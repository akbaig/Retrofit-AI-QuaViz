"""Causal per-minute features.

Every row describes one decision time. A row for the 1-min window starting at t can only be acted on at
t + 1 min (``decision_time``), and every value in it is computed from rows with window_start <= t, so an alert
raised at ``decision_time`` uses only data available at that time. Static asset information (``assets.csv``) is
allowed. ``tests/test_features_causal.py`` checks this by truncating the data.

Signals are scaled by nameplate ratings so that pumps of different size share one feature space; the scaling uses
no pump-curve physics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset, ts

CH = {"n": "motor_speed_rpm", "i": "motor_current_a", "pw": "motor_power_kw", "ps": "suction_pressure_kpa",
      "pd": "discharge_pressure_kpa", "q": "flow_m3_h", "tf": "fluid_temp_c", "bt": "bearing_temp_c",
      "wt": "motor_winding_temp_c", "v": "vibration_rms_mm_s", "a": "vibration_accel_rms_g",
      "c": "vibration_crest_factor", "vp": "valve_position_pct", "amb": "ambient_temp_c"}
WINDOWS = (30, 120, 360)
SETTLE_MIN = 10
CONTEXT = {"duty_demand_pct": "duty", "supply_water_temp_c": "supply_t", "sea_temp_c": "sea_t",
           "sea_state_hs_m": "hs", "suction_static_head_m": "level", "ambient_temp_c": "amb_ctx"}
PROFILES = ["SHIP_SW_COOLING", "SHIP_LT_FRESHWATER", "INDUSTRIAL_CW", "PROCESS_TRANSFER"]
CONTROLS = ["VFD_TEMPERATURE", "VFD_PRESSURE", "FLOW_VALVE", "FIXED_SPEED"]
TIERS = ["T0_LEGACY", "T1_RETROFIT", "T2_FULL"]


def scales(a: pd.Series) -> dict:
    """Divisor per channel from the nameplate (1 where no rating applies)."""
    head_kpa = 9.81 * float(a["pump_rated_head_m"])
    return {"n": float(a["motor_rated_rpm"]), "i": float(a["motor_rated_current_a"]), "pw": float(a["motor_rated_kw"]),
            "ps": head_kpa, "pd": head_kpa, "q": float(a["pump_rated_flow_m3_h"]),
            "v": float(a["vibration_alarm_mm_s"]) if pd.notna(a["vibration_alarm_mm_s"]) else 1.0, "vp": 100.0}


def _run_length(flag: np.ndarray) -> np.ndarray:
    """Consecutive True count ending at each position."""
    out = np.zeros(len(flag), dtype=np.int32)
    c = 0
    for j, x in enumerate(flag):
        c = c + 1 if x else 0
        out[j] = c
    return out


def minute_frame(ds: Dataset, asset_id: str) -> pd.DataFrame:
    """Complete 1-min grid over the asset window with scaled channel means, sample counts, mode and context."""
    a = ds.asset(asset_id)
    grid = pd.date_range(ts(pd.Series([a["window_start_utc"]]))[0], ts(pd.Series([a["window_end_utc"]]))[0],
                         freq="1min", inclusive="left")
    f = ds.features[ds.features["asset_id"] == asset_id].set_index("window_start_utc").reindex(grid)
    out = pd.DataFrame(index=grid)
    out.index.name = "window_start_utc"
    out["asset_id"] = asset_id
    out["decision_time"] = grid + pd.Timedelta(minutes=1)
    out["rows"] = f["rows"].fillna(0).astype(np.int16).values
    mode = f["operating_mode"].astype(object).where(f["operating_mode"].notna(), "NO_DATA")
    out["mode"] = mode.values
    out["running"] = ((mode == "RUNNING") & (f["running_fraction"].fillna(0) >= 0.99)).values
    out["run_age_min"] = _run_length(out["running"].values)
    out["steady"] = out["run_age_min"] >= SETTLE_MIN
    avail = set(str(a["channels_available"]).split("|"))
    sc = scales(a)
    for k, col in CH.items():
        m = f"{col}__mean"
        if col not in avail or m not in f:
            continue
        div = sc.get(k, 1.0)
        out[k] = (f[m] / div).values
        out[k + "_raw"] = f[m].values
        out[k + "_mstd"] = (f[f"{col}__std"] / div).values
        out[k + "_cnt"] = f[f"{col}__count"].fillna(0).values
    c = ds.context[ds.context["asset_id"] == asset_id].set_index("timestamp_utc").reindex(grid)
    for col, k in CONTEXT.items():
        out[k] = c[col].values if col in c else np.nan
    out["schedule"] = c["schedule_state"].astype(object).values if "schedule_state" in c else None
    return out


def _rolling_slope_per_h(x: pd.Series, w: int, min_periods: int) -> pd.Series:
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index).where(x.notna())
    xv = x.where(t.notna())
    r = lambda s: s.rolling(w, min_periods=1).sum()
    n = x.notna().astype(float).rolling(w, min_periods=1).sum()
    st, sx, stx, stt = r(t.fillna(0)), r(xv.fillna(0)), r((t * xv).fillna(0)), r((t * t).fillna(0))
    den = n * stt - st * st
    slope = (n * stx - st * sx) / den.where(den > 0)
    return (slope * 60.0).where(n >= min_periods)


def _minutes_since(mask_times: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since the last time in ``mask_times`` (NaN before the first)."""
    last = mask_times.reindex(idx).ffill()
    return ((idx - pd.DatetimeIndex(last)).total_seconds() / 60.0).to_numpy()


def window_features(fr: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    idx = fr.index
    out = {"asset_id": fr["asset_id"].values, "decision_time": fr["decision_time"],
           "running": fr["running"].values, "steady": fr["steady"].values, "run_age_min": fr["run_age_min"].values}
    rows = fr["rows"].astype(float)
    out["rows"] = rows.values
    for w in (10, 60):
        out[f"gap_frac_{w}"] = (1.0 - rows.rolling(w, min_periods=1).sum() / (60.0 * w)).values
    out["no_data_run_min"] = _run_length((rows == 0).values)
    for k in CH:
        if k not in fr:
            continue
        raw = fr[k + "_raw"]
        x = fr[k].where(fr["steady"])
        for w in windows:
            mp = max(3, w // 10)
            roll = x.rolling(w, min_periods=mp)
            out[f"{k}_m{w}"] = roll.mean().values
            out[f"{k}_s{w}"] = roll.std().values
            out[f"{k}_slope{w}"] = _rolling_slope_per_h(x, w, mp).values
        out[f"{k}_d_short_long"] = out[f"{k}_m{windows[0]}"] - out[f"{k}_m{windows[-1]}"]
        sample = raw.dropna()
        out[f"{k}_age_min"] = _minutes_since(pd.Series(sample.index, index=sample.index), idx)
        changed = sample.diff().ne(0)
        out[f"{k}_flat_min"] = _minutes_since(pd.Series(sample.index[changed.values], index=sample.index[changed.values]), idx)
        ms = fr[k + "_mstd"].where(fr["steady"])
        short = ms.rolling(30, min_periods=5).median()
        long = ms.rolling(1440, min_periods=60).median()
        out[f"{k}_noise_ratio"] = (short / long.where(long > 0)).values
    duty = fr["duty"]
    out["duty"] = duty.values
    out["duty_change_30"] = (duty - duty.shift(30)).values
    for k in ("supply_t", "sea_t", "hs", "level"):
        out[k] = fr[k].values
    return pd.DataFrame(out, index=idx)


def static_features(a: pd.Series) -> dict:
    avail = set(str(a["channels_available"]).split("|"))
    d = {f"profile_{p}": float(a["profile"] == p) for p in PROFILES}
    d |= {f"control_{c}": float(a["control"] == c) for c in CONTROLS}
    d |= {f"tier_{t}": float(a["instrumentation_tier"] == t) for t in TIERS}
    d |= {f"has_{k}": float(col in avail) for k, col in CH.items()}
    d["wireless"] = float(pd.notna(a.get("wireless_update_s")))
    return d


def asset_features(ds: Dataset, asset_id: str, stride_min: int = 1) -> pd.DataFrame:
    fr = minute_frame(ds, asset_id)
    wf = window_features(fr)
    for k, v in static_features(ds.asset(asset_id)).items():
        wf[k] = v
    if stride_min > 1:
        wf = wf.iloc[::stride_min]
    return wf


def build_features(ds: Dataset, asset_ids=None, stride_min: int = 1) -> pd.DataFrame:
    ids = asset_ids if asset_ids is not None else ds.ids()
    return pd.concat([asset_features(ds, a, stride_min) for a in ids], ignore_index=False)
