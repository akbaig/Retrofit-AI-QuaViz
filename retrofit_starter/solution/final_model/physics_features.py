"""Causal, asset-normalized pump features derived only from public measurements.

``add_physics_features`` returns a new numeric frame, with the input index and
``phys__`` column names. It never reads files, labels, simulator configurations,
latent states, identifiers, or future rows. Supply one asset's rows in increasing
time order; minute aggregates must be timestamped at their completion time.

Inputs are the public ``assets.csv`` row and minute telemetry columns named
``<channel>__mean/std/min/max/count``. Operating context may be joined without a
prefix, or with ``ctx__`` / ``context__``. Bounded, past-only holding of sparse
observations belongs to the caller, which should retain sample-age features.

Nominal engineering assumptions are deliberately shared across assets: 25 C
design supply for temperature control, a 25% equal-percentage valve contribution,
2.5 m/s suction velocity at BEP, 28 K bearing rise, and 75 K winding rise. These
approximate model-based residuals are NOT healthy truth or calibrated diagnoses.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np
import pandas as pd

G = 9.80665
P_ATM = 101_325.0


def _rating(asset: Mapping, name: str, default: float = np.nan) -> float:
    try:
        value = float(asset.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _column(frame: pd.DataFrame, *names: str) -> pd.Series:
    """Select the first available column, preserving missing individual readings."""
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _context(frame: pd.DataFrame, name: str) -> pd.Series:
    return _column(frame, "ctx__" + name, "context__" + name, name)


def _safe_div(numerator, denominator, floor: float = 1e-8):
    """Undefined denominators remain NaN, rather than looking like healthy zero."""
    return numerator / np.where(np.abs(denominator) > floor, denominator, np.nan)


def _water_density(temp_c, salinity=0.0):
    t = np.clip(temp_c, -5.0, 100.0)
    return 999.84 + 0.0679*t - 0.009096*t*t + 1e-4*t**3 - 1e-6*t**4 + 0.78*salinity


def add_physics_features(frame: pd.DataFrame, asset: Mapping) -> pd.DataFrame:
    """Return deterministic numeric physics features; missing evidence stays NaN.

    The only stateful transforms are forward exponential thermal lags. Processing
    a prefix therefore produces the same features as the corresponding prefix of
    a full run. Pass complete one-minute grids for meaningful lag constants.
    """
    def observed(channel):
        return _column(frame, channel + "__mean")

    def context_or_observed(context_name, observation_name):
        context_value = _context(frame, context_name)
        return context_value.combine_first(observed(observation_name))

    qb = _rating(asset, "pump_rated_flow_m3_h")
    hb = _rating(asset, "pump_rated_head_m")
    hshut = _rating(asset, "pump_shutoff_head_m")
    h0 = hshut / hb if hb > 0 else np.nan
    qdesign = _rating(asset, "design_flow_m3_h")
    hdesign = _rating(asset, "design_head_m")
    hstatic = _rating(asset, "static_head_m")
    eta_bep = _rating(asset, "pump_bep_efficiency")
    motor_kw = _rating(asset, "motor_rated_kw")
    current_rated = _rating(asset, "motor_rated_current_a")
    rpm_rated = _rating(asset, "motor_rated_rpm")
    voltage = _rating(asset, "supply_voltage_v")
    npsh_bep = _rating(asset, "npsh3_at_bep_m")
    vib_alarm = _rating(asset, "vibration_alarm_mm_s")
    control = str(asset.get("control", ""))

    current = observed("motor_current_a")
    power = observed("motor_power_kw")
    rpm = observed("motor_speed_rpm")
    running = _column(frame, "running_fraction")
    if "running_fraction" not in frame and "operating_mode" in frame:
        running = frame["operating_mode"].isin(["RUNNING", "STARTING"]).astype(float)
    speed = rpm / rpm_rated
    # DOL equipment has known fixed speed while running, not a missing healthy
    # sensor reading. Do not make this substitution for a variable-speed drive.
    if control in ("FIXED_SPEED", "FLOW_VALVE"):
        speed = speed.combine_first(running)
    active_speed = speed.where((speed > 0.2) & (running >= 0.5))

    fluid = observed("fluid_temp_c")
    supply = context_or_observed("supply_water_temp_c", "fluid_temp_c")
    ambient = context_or_observed("ambient_temp_c", "ambient_temp_c")
    salinity_match = re.search(r"([0-9.]+)\s*PSU", str(asset.get("fluid", "")), re.I)
    salinity = float(salinity_match.group(1)) if salinity_match else 0.0
    rho = _water_density(supply, salinity)
    pressure_scale = rho * G * hb / 1000.0
    suction = observed("suction_pressure_kpa")
    discharge = observed("discharge_pressure_kpa")
    head = (discharge - suction) * 1000.0 / (rho * G)
    flow = observed("flow_m3_h")
    q = flow / qb
    q_at_speed = q / active_speed
    healthy_head_at_observed_q = hb * (h0 * active_speed**2 - (h0-1.0)*q**2)
    nominal_curve_q2 = (h0*active_speed**2 - head/hb) / (h0-1.0)
    inferred_flow = qb * np.sqrt(nominal_curve_q2.where(nominal_curve_q2 >= 0.0))
    # Separate inferred and measured channels; head/curve residual never uses
    # inferred flow, which would make it identically zero by construction.
    hydraulic_flow = flow.combine_first(inferred_flow)
    hydraulic_q = hydraulic_flow / qb / active_speed

    demand = _context(frame, "duty_demand_pct") / 100.0
    k_design = (hdesign-hstatic) / qdesign**2 if qdesign > 0 else np.nan
    nominal_k = pd.Series(k_design, index=frame.index, dtype=float)
    expected_speed = pd.Series(1.0, index=frame.index, dtype=float)
    required_flow = pd.Series(np.nan, index=frame.index, dtype=float)
    setpoint_column = next((name for name in ("ctx__controlled_setpoint", "context__controlled_setpoint", "controlled_setpoint") if name in frame), None)
    if setpoint_column is None:
        setpoint_number = pd.Series(np.nan, index=frame.index, dtype=float)
    else:
        setpoint_number = pd.to_numeric(frame[setpoint_column].astype(str).str.extract(r"([-+]?[0-9]+(?:\.[0-9]+)?)", expand=False), errors="coerce")

    if control == "VFD_TEMPERATURE":
        t_set = setpoint_number.fillna(36.0)
        required_flow = (demand*(t_set-25.0)/(t_set-supply).clip(lower=3.0)).clip(0.4, 1.15)*qdesign
        expected_speed = np.sqrt((hstatic + (k_design + hb*(h0-1.0)/qb**2)*required_flow**2)/(hb*h0))
    elif control == "VFD_PRESSURE":
        valve_command = (0.35 + 0.65*demand).clip(0.2, 1.0)
        nominal_k = k_design*(0.75 + 0.25*50.0**(2.0*(0.8-valve_command)))
        dp_head = (setpoint_number*1000.0/(rho*G)).fillna(0.9*hdesign)
        required_flow = np.sqrt((dp_head-hstatic).clip(lower=0.0)/nominal_k)
        expected_speed = np.sqrt((dp_head + hb*(h0-1.0)*(required_flow/qb)**2)/(hb*h0))
    elif control == "FLOW_VALVE":
        required_flow = demand.clip(0.2, 1.1)*qdesign
        nominal_k = (hshut-hstatic)/required_flow**2 - hb*(h0-1.0)/qb**2

    if control.startswith("VFD"):
        expected_speed = expected_speed.clip(_rating(asset, "min_speed_ratio", 0.5), 1.0)
    expected_flow = np.sqrt((hb*h0*active_speed**2-hstatic).clip(lower=0.0)/(hb*(h0-1.0)/qb**2 + nominal_k))
    # Hydraulic power is an observed consistency relation. It is not filled from
    # curve-inferred flow when a measured flow channel is absent.
    hydraulic_kw = rho*G*(flow/3600.0)*head/1000.0
    eta_speed = 1.0-(1.0-eta_bep)*active_speed**-0.1
    eta_curve = eta_speed*q_at_speed*(2.0-q_at_speed)
    expected_shaft_kw = _safe_div(hydraulic_kw, eta_curve.where(eta_curve > 0.05))

    vapor_pa = 133.322*10.0**(8.07131-1730.63/(233.426+supply))
    static_npsh = (suction*1000.0+P_ATM-vapor_pa)/(rho*G)
    dynamic_suction_head = (2.5*hydraulic_flow/qb)**2/(2.0*G)
    npsha = static_npsh + dynamic_suction_head
    npshr = npsh_bep*active_speed**2*(0.55 + 0.45*hydraulic_q.clip(0.0, 2.0)**2.6)
    suction_level = _context(frame, "suction_static_head_m")

    velocity = observed("vibration_rms_mm_s")
    acceleration = observed("vibration_accel_rms_g")
    crest = observed("vibration_crest_factor")
    current_pu = current/current_rated
    power_pu = power/motor_kw
    # A load proxy is used only in broad thermal normalization, never as a label.
    load_proxy = (0.94*power_pu).combine_first(current_pu).clip(0.0, 1.5)
    thermal_base = 0.6*ambient + 0.4*fluid
    bearing_rise = observed("bearing_temp_c") - thermal_base
    nominal_bearing_rise = 28.0*speed.clip(lower=0.0)**1.3*(0.75+0.25*load_proxy.clip(upper=1.2))*running
    nominal_bearing_temp = (thermal_base + nominal_bearing_rise).ewm(alpha=1.0-np.exp(-60.0/1500.0), adjust=False).mean()
    nominal_winding_temp = (ambient + 75.0*(0.25+0.75*load_proxy**2)*running).ewm(alpha=1.0-np.exp(-60.0/2400.0), adjust=False).mean()

    values = {
        "speed_pu": speed,
        "current_pu": current_pu,
        "power_pu": power_pu,
        "current_speed_cubed": current_pu/active_speed**3,
        "power_speed_cubed": power_pu/active_speed**3,
        "electrical_power_factor": power*1000.0/(np.sqrt(3.0)*voltage*current.where(current > 0.01*current_rated)),
        "speed_command_residual": active_speed-expected_speed,
        "flow_pu": flow/qdesign,
        "flow_bep_at_speed": q_at_speed,
        "flow_required_residual": (flow-required_flow)/qdesign,
        "flow_nominal_system_residual": (flow-expected_flow)/qdesign,
        "curve_inferred_flow_pu": inferred_flow/qdesign,
        "curve_inferred_flow_required_residual": (inferred_flow-required_flow)/qdesign,
        "head_pu": head/hb,
        "head_speed_squared": head/(hb*active_speed**2),
        "head_curve_residual": (head-healthy_head_at_observed_q)/hb,
        "head_curve_factor": _safe_div(head, healthy_head_at_observed_q, 1.0),
        "suction_pu": suction/pressure_scale,
        "discharge_pu": discharge/pressure_scale,
        "hydraulic_efficiency": _safe_div(hydraulic_kw, power, 0.01),
        "power_shaft_residual_pu": (power-expected_shaft_kw)/motor_kw,
        "system_resistance_factor": (head-hstatic)/(nominal_k*flow.where(flow > 0.1*qdesign)**2),
        "npsha_static_pu": static_npsh/npsh_bep,
        "npsh_margin": _safe_div(npsha, npshr, 0.01),
        "suction_level_loss_pu": (suction_level-suction*1000.0/(rho*G)-dynamic_suction_head)/hb,
        "bearing_rise_k": bearing_rise,
        "bearing_rise_speed_normalized": bearing_rise/active_speed**1.3,
        "bearing_thermal_residual_k": observed("bearing_temp_c")-nominal_bearing_temp,
        "winding_rise_k": observed("motor_winding_temp_c")-ambient,
        "winding_thermal_residual_k": observed("motor_winding_temp_c")-nominal_winding_temp,
        "fluid_supply_excess_k": fluid-supply,
        "vibration_alarm_fraction": velocity/vib_alarm,
        "vibration_speed_squared": velocity/active_speed**2,
        "acceleration_speed_normalized": acceleration/active_speed,
        "acceleration_velocity_ratio": _safe_div(acceleration, velocity, 0.1),
        "acceleration_rotational_ratio": _safe_div(acceleration*G, velocity/1000.0*2.0*np.pi*(active_speed*rpm_rated/60.0), 0.01),
        "crest_excess": crest-3.2,
        "acceleration_crest_product": acceleration*(crest-3.2),
        "velocity_crest_ratio": _safe_div(velocity, crest, 1.0),
        "hydraulic_off_bep": (hydraulic_q-1.0).abs(),
        "valve_demand_residual": observed("valve_position_pct")/100.0-(0.35+0.65*demand).clip(0.2, 1.0) if control == "VFD_PRESSURE" else pd.Series(np.nan, index=frame.index),
        "suction_std_pu": _column(frame, "suction_pressure_kpa__std")/pressure_scale,
        "discharge_std_pu": _column(frame, "discharge_pressure_kpa__std")/pressure_scale,
        "pressure_std_ratio": _safe_div(_column(frame, "suction_pressure_kpa__std"), _column(frame, "discharge_pressure_kpa__std"), 0.05),
        "current_std_pu": _column(frame, "motor_current_a__std")/current_rated,
        "flow_std_pu": _column(frame, "flow_m3_h__std")/qdesign,
        "current_range_pu": (_column(frame, "motor_current_a__max")-_column(frame, "motor_current_a__min"))/current_rated,
        "discharge_range_pu": (_column(frame, "discharge_pressure_kpa__max")-_column(frame, "discharge_pressure_kpa__min"))/pressure_scale,
    }
    result = pd.DataFrame({"phys__"+key: value for key, value in values.items()}, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def extract_snapshot_features(samples, sample_rate_hz: float, rpm: float, asset: Mapping) -> dict[str, float]:
    """Observed waveform evidence for a past-only snapshot join, with no labels.

    Order-band velocity resolves misalignment's 2x/3x components from imbalance's
    1x. Envelope BPFO concentration resolves repetitive bearing impacts from
    broadband hydraulic excitation. A one-second record limits resolution; bands
    allow frequency leakage and the documented small rolling-element slip.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    names = ("rms_g", "crest", "kurtosis", "velocity_1x", "velocity_2x", "velocity_3x", "velocity_vane", "harmonic_2x_1x", "harmonic_3x_1x", "high_frequency_fraction", "envelope_bpfo_fraction", "envelope_bpfi_fraction", "envelope_bpfo_prominence")
    out = {"snapshot__"+name: np.nan for name in names}
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or len(x) < 128 or not np.isfinite(x).all() or not np.isfinite(rpm) or rpm < 60 or sample_rate_hz <= 0:
        return out
    x = x-x.mean()
    n = len(x)
    fs = float(sample_rate_hz)
    fr = float(rpm)/60.0
    window = np.hanning(n)
    frequency = np.fft.rfftfreq(n, 1.0/fs)
    power = np.abs(np.fft.rfft(x*window))**2 * 2.0/(n*np.sum(window**2))
    power[0] *= 0.5
    if n % 2 == 0:
        power[-1] *= 0.5
    rms = np.sqrt(np.mean(x*x))
    out["snapshot__rms_g"] = rms
    out["snapshot__crest"] = float(np.max(np.abs(x))/max(rms, 1e-9))
    out["snapshot__kurtosis"] = float(np.mean(x**4)/max(rms**4, 1e-18))
    line_width = max(2.5*fs/n, 0.04*fr)

    def line_velocity(order):
        center = order*fr
        band = (np.abs(frequency-center) <= line_width) & (frequency > 1.0)
        # Integrate the acceleration spectrum: this is invariant to rpm scaling.
        return float(np.sqrt(np.sum(power[band]*(1000.0*G/(2.0*np.pi*frequency[band]))**2))) if band.any() else np.nan

    v1, v2, v3 = (line_velocity(order) for order in (1.0, 2.0, 3.0))
    out.update({"snapshot__velocity_1x": v1, "snapshot__velocity_2x": v2, "snapshot__velocity_3x": v3,
                "snapshot__velocity_vane": line_velocity(_rating(asset, "impeller_vanes", 6)),
                "snapshot__harmonic_2x_1x": v2/max(v1, 0.01), "snapshot__harmonic_3x_1x": v3/max(v1, 0.01),
                "snapshot__high_frequency_fraction": float(power[frequency >= 1000].sum()/max(power.sum(), 1e-12))})
    high = min(4000.0, 0.9*fs/2.0)
    low = min(1800.0, 0.5*high)
    filtered = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=fs, output="sos"), x)
    envelope = np.abs(hilbert(filtered))
    envelope -= envelope.mean()
    envelope_power = np.abs(np.fft.rfft(envelope*window))**2
    envelope_total = envelope_power[(frequency >= 10) & (frequency <= min(1000.0, fs/2))].sum()
    for label in ("bpfo", "bpfi"):
        order = _rating(asset, "bearing_"+label+"_order")
        harmonic_bands = np.zeros(len(frequency), dtype=bool)
        adjacent_bands = np.zeros(len(frequency), dtype=bool)
        for multiple in (1, 2, 3):
            center = order*fr*multiple
            width = max(3.0*fs/n, center*0.025)
            harmonic_bands |= np.abs(frequency-center) <= width
            adjacent_bands |= (np.abs(frequency-center) > width) & (np.abs(frequency-center) <= 3*width)
        energy = envelope_power[harmonic_bands].sum()
        out["snapshot__envelope_"+label+"_fraction"] = float(energy/max(envelope_total, 1e-12)) if np.isfinite(order) else np.nan
        if label == "bpfo":
            out["snapshot__envelope_bpfo_prominence"] = float(envelope_power[harmonic_bands].mean()/max(envelope_power[adjacent_bands].mean(), 1e-12)) if harmonic_bands.any() and adjacent_bands.any() else np.nan
    return out
