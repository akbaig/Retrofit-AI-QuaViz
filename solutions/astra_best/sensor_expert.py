"""Shared, channel-anonymous sensor fault models over causal evidence.

Interface::
    expert = SensorExpert().fit(training_X, training_meta, training_events)
    scores = expert.predict(X, meta)  # four columns, max over observed channels

The classifier is shared across channels: no channel names, identities, ratings,
calendar timestamps, profiles, or fault schedules enter its feature matrix.
Asset IDs only isolate causal histories and assign training targets. Channel
names only select observations and correctly align event.sensor_tag supervision.
Optional ``residual__<channel>`` features may come from training-fitted healthy
regressors; the caller must supply the same transformation at inference.
Supply the available history prefix for each asset at prediction time, not only
the latest row; the component derives references from previous measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

SENSOR_TYPES = ("SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT", "SENSOR_NOISE_SPIKES")

# These select observed physical consistency relations, not class/type identities.
# They are converted to the same historical-noise-normalized generic columns.
COHERENCE = {
    "motor_current_a": "phys__electrical_power_factor",
    "motor_power_kw": "phys__power_shaft_residual_pu",
    "motor_speed_rpm": "phys__speed_command_residual",
    "suction_pressure_kpa": "phys__head_curve_residual",
    "discharge_pressure_kpa": "phys__head_curve_residual",
    "flow_m3_h": "phys__flow_required_residual",
    "fluid_temp_c": "phys__fluid_supply_excess_k",
    "bearing_temp_c": "phys__bearing_thermal_residual_k",
    "motor_winding_temp_c": "phys__winding_thermal_residual_k",
    "vibration_rms_mm_s": "phys__acceleration_velocity_ratio",
    "vibration_accel_rms_g": "phys__acceleration_velocity_ratio",
    "valve_position_pct": "phys__valve_demand_residual",
}


def _col(x, name):
    return pd.to_numeric(x[name], errors="coerce").astype(float) if name in x else pd.Series(np.nan, index=x.index)


def _ratio(a, b, limit=100.0):
    denominator = np.where(np.abs(b) > 1e-10, b, np.nan)
    return np.clip(a / denominator, -limit, limit)


def channel_names(x):
    return sorted(c.removeprefix("level__") for c in x if c.startswith("level__"))


def make_channel_targets(meta, events, channel):
    """Match supervision to the affected sensor, never label every tag positive.

    A physically active but not-yet-detectable or unobservable target fault is
    unknown (-1). Its visible period begins at max(onset, detectable). Different
    fault types on the channel remain informative negative examples.
    """
    n = len(meta)
    y = pd.DataFrame(0, index=range(n), columns=SENSOR_TYPES, dtype=np.int8)
    weights = pd.DataFrame(1.0, index=range(n), columns=SENSOR_TYPES)
    t = pd.to_datetime(meta.available_utc, utc=True)
    for _, event in events.iterrows():
        if event.fault_type not in SENSOR_TYPES or str(event.get("sensor_tag", "")) != channel:
            continue
        onset = pd.to_datetime(event.onset_utc, utc=True)
        end = pd.to_datetime(event.end_utc, utc=True)
        detected = pd.to_datetime(event.detectable_utc, utc=True)
        active = (meta.asset_id.eq(event.asset_id) & t.ge(onset) & (t.lt(end) if pd.notna(end) else True)).to_numpy()
        if str(event.observable).lower() not in ("true", "1"):
            y.loc[active, event.fault_type] = -1
            continue
        visible = active & t.ge(max(onset, detected) if pd.notna(detected) else onset).to_numpy()
        y.loc[active & ~visible, event.fault_type] = -1
        y.loc[visible, event.fault_type] = 1
        if visible.any():
            weights.loc[visible, event.fault_type] = 1000.0 / visible.sum()
    return y, weights


def _one_history(x, channel):
    """All references are shifted before expanding/rolling; never future fitted."""
    level = _col(x, "level__" + channel)
    fresh = _col(x, "fresh__" + channel).fillna(0).gt(0)
    age = _col(x, "age__" + channel)
    observed = level.where(fresh)
    previous_observation = observed.ffill().shift(1)
    changes = (observed - previous_observation).where(fresh)
    count = _col(x, "count__" + channel) * 60.0
    within_std = _col(x, "std__" + channel).where(fresh & count.ge(2))
    steady = _col(x, "obs__steady").fillna(1).gt(0)

    # The lower quantile is resistant to intermittent noise/spikes. A 60-minute
    # delay prevents an onset from immediately redefining its own reference.
    noise_reference = within_std.where(steady).shift(60).expanding(min_periods=20).quantile(.25)
    step_reference = changes.abs().where(steady).shift(60).expanding(min_periods=20).quantile(.5)
    mean_noise_reference = noise_reference / np.sqrt(count.clip(lower=1))
    scale = pd.concat([step_reference, mean_noise_reference], axis=1).max(axis=1)
    scale = scale.where(scale > 1e-10)

    history = observed.shift(60)
    q25 = history.expanding(min_periods=30).quantile(.25)
    q50 = history.expanding(min_periods=30).quantile(.5)
    q75 = history.expanding(min_periods=30).quantile(.75)
    operating_spread = pd.concat([q75-q25, 4.0*scale], axis=1).max(axis=1)
    mean10 = level.rolling(10, min_periods=2).mean()
    mean60 = level.rolling(60, min_periods=5).mean()
    std10 = level.rolling(10, min_periods=3).std()
    std60 = level.rolling(60, min_periods=10).std()
    d10, d60 = level-level.shift(10), level-level.shift(60)
    abs_changes = changes.abs()
    max_step10 = abs_changes.rolling(10, min_periods=1).max()
    sum_step60 = abs_changes.rolling(60, min_periods=3).sum()
    count_step60 = changes.rolling(60, min_periods=1).count()
    span60 = level.rolling(60, min_periods=3).max()-level.rolling(60, min_periods=3).min()

    data = {
        "noise_ratio": _ratio(within_std, noise_reference),
        "noise_ratio10": _ratio(within_std.rolling(10, min_periods=2).mean(), noise_reference),
        "noise_ratio60": _ratio(within_std.rolling(60, min_periods=5).mean(), noise_reference),
        "noise_max_ratio10": _ratio(within_std.rolling(10, min_periods=2).max(), noise_reference),
        "noise_ratio10_60": _ratio(within_std.rolling(10, min_periods=2).mean(), within_std.rolling(60, min_periods=5).mean()),
        "std_span_ratio": _ratio(_col(x, "range__"+channel), within_std),
        "step_z": _ratio(changes, scale),
        "abs_step_z": _ratio(abs_changes, scale),
        "abs_step_mean10_z": _ratio(abs_changes.rolling(10, min_periods=2).mean(), scale),
        "abs_step_mean60_z": _ratio(abs_changes.rolling(60, min_periods=5).mean(), scale),
        "max_step10_z": _ratio(max_step10, scale),
        "level_std10_z": _ratio(std10, scale),
        "level_std60_z": _ratio(std60, scale),
        "delta10_z": _ratio(d10, scale),
        "delta60_z": _ratio(d60, scale),
        "abs_delta10_z": np.abs(_ratio(d10, scale)),
        "abs_delta60_z": np.abs(_ratio(d60, scale)),
        "mean10_60_z": _ratio(mean10-mean60, scale),
        "mean10_60_abs_z": np.abs(_ratio(mean10-mean60, scale)),
        "historical_offset": _ratio(level-q50, operating_spread),
        "historical_offset_abs": np.abs(_ratio(level-q50, operating_spread)),
        "mean60_historical_offset": _ratio(mean60-q50, operating_spread),
        "mean60_historical_offset_abs": np.abs(_ratio(mean60-q50, operating_spread)),
        "trend_directionality": _ratio(np.abs(d60), sum_step60),
        "jump_fraction": _ratio(max_step10, span60+scale),
        "flat_minutes": _col(x, "flat_minutes__"+channel).clip(upper=240),
        "equal_updates": _col(x, "equal_updates__"+channel).clip(upper=240),
        "zero_variance": (within_std == 0).where(within_std.notna()).astype(float),
        "flat_fraction60": changes.eq(0).where(changes.notna()).rolling(60, min_periods=3).mean(),
        "observed_changes60": count_step60,
        "running": _col(x, "obs__running"),
        "steady": _col(x, "obs__steady"),
    }
    # Context-change magnitudes help distinguish plant transitions from isolated
    # sensor departures without identifying a plant or operating temperature.
    context = pd.DataFrame({
        "demand": _col(x, "ctx__duty_demand_pct")/100.0,
        "supply": _col(x, "ctx__supply_water_temp_c")/30.0,
        "ambient": _col(x, "ctx__ambient_temp_c")/30.0,
        "speed": _col(x, "level__motor_speed_rpm"),
    })
    for window in (10, 60):
        data[f"context_change{window}"] = (context-context.shift(window)).abs().max(axis=1)

    coherence = _col(x, COHERENCE.get(channel, "__unavailable__"))
    ch = coherence.shift(60)
    chmedian = ch.expanding(min_periods=30).median()
    chspread = ch.expanding(min_periods=30).quantile(.75)-ch.expanding(min_periods=30).quantile(.25)
    chstep = coherence.diff().abs().shift(60).expanding(min_periods=30).median()
    chscale = pd.concat([chspread, chstep*4], axis=1).max(axis=1)
    data["coherence_offset"] = _ratio(coherence-chmedian, chscale)
    data["coherence_offset_abs"] = np.abs(data["coherence_offset"])
    data["coherence_delta60"] = _ratio(coherence-coherence.shift(60), chscale)

    # An absolute consistency check can detect a pre-existing offset before
    # there is local history. The common motor PF/load curve is approximate;
    # it supplies classifier evidence and never directly raises an alert.
    if channel in ("motor_current_a", "motor_power_kw"):
        load = (.94*_col(x, "phys__power_pu")).clip(0, 1.5)
        expected_pf = .85*(1-np.exp(-load/.3))/(1-np.exp(-1/.3))
        absolute_coherence = (_col(x, "phys__electrical_power_factor")-expected_pf)/.08
    else:
        coherence_scales = {"motor_speed_rpm": .02, "suction_pressure_kpa": .03,
            "discharge_pressure_kpa": .03, "flow_m3_h": .05,
            "fluid_temp_c": 1., "bearing_temp_c": 5.,
            "motor_winding_temp_c": 10., "valve_position_pct": .05}
        absolute_coherence = coherence/coherence_scales[channel] if channel in coherence_scales else pd.Series(np.nan, index=x.index)
    data["absolute_coherence_z"] = absolute_coherence.clip(-100, 100)
    data["absolute_coherence_abs_z"] = data["absolute_coherence_z"].abs()
    coherence_error_history = data["absolute_coherence_z"].shift(60)
    coherence_error_anchor = coherence_error_history.expanding(min_periods=30).median()
    coherence_error_spread = (coherence_error_history.expanding(min_periods=30).quantile(.75)
                              - coherence_error_history.expanding(min_periods=30).quantile(.25)).clip(lower=.05)
    data["consistency_departure"] = _ratio(data["absolute_coherence_z"]-coherence_error_anchor, coherence_error_spread)
    data["consistency_departure_abs"] = np.abs(data["consistency_departure"])
    data["consistency_slope60"] = _ratio(data["absolute_coherence_z"]-data["absolute_coherence_z"].shift(60), coherence_error_spread)

    residual = _col(x, "residual__"+channel)
    data["learned_residual"] = residual
    data["learned_residual_abs"] = residual.abs()
    data["learned_residual_mean60"] = residual.rolling(60, min_periods=2).mean()
    data["learned_residual_mean60_abs"] = np.abs(data["learned_residual_mean60"])
    data["learned_residual_delta10"] = residual-residual.shift(10)
    data["learned_residual_delta60"] = residual-residual.shift(60)
    residual_history = residual.shift(60)
    residual_anchor = residual_history.expanding(min_periods=30).median()
    residual_spread = (residual_history.expanding(min_periods=30).quantile(.75)
                       - residual_history.expanding(min_periods=30).quantile(.25)).clip(lower=1.0)
    data["residual_departure"] = _ratio(residual-residual_anchor, residual_spread)
    data["residual_departure_abs"] = np.abs(data["residual_departure"])
    data["residual_departure_mean60"] = pd.Series(data["residual_departure"], index=x.index).rolling(60, min_periods=5).mean()
    data["residual_trend_directionality"] = _ratio((residual-residual.shift(60)).abs(), residual.diff().abs().rolling(60, min_periods=5).sum()+.25)
    other_residuals = [name for name in x if name.startswith("residual__") and name != "residual__"+channel]
    if other_residuals:
        peers = x[other_residuals].abs().median(axis=1)
        data["peer_residual_median"] = peers
        data["residual_dominance"] = residual.abs()/(peers+.25)
    else:
        data["peer_residual_median"] = np.nan
        data["residual_dominance"] = np.nan
    result = pd.DataFrame(data, index=x.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    eligible = level.notna() & age.le(10) & (_col(x, "obs__gap").fillna(0) == 0)
    return result, eligible.to_numpy()


def channel_features(x, meta, channel):
    """Return causal generic features and an observed-channel eligibility mask."""
    if len(x) != len(meta):
        raise ValueError("X and metadata must have the same row order and length")
    pieces = []
    masks = np.zeros(len(x), dtype=bool)
    times = pd.to_datetime(meta.available_utc, utc=True)
    for ids in meta.groupby("asset_id", sort=False).indices.values():
        if not times.iloc[ids].is_monotonic_increasing:
            raise ValueError("Each asset must be ordered chronologically")
        frame = x.iloc[ids].copy()
        frame.index = ids
        values, eligible = _one_history(frame, channel)
        pieces.append(values)
        masks[ids] = eligible
    if not pieces:
        return pd.DataFrame(index=x.index), masks
    result = pd.concat(pieces).sort_index()
    result.index = x.index
    return result, masks


@dataclass
class SensorExpert:
    """Four shared channel-anonymous classifiers with max-channel pooling."""
    seed: int = 20261107
    max_positive_per_type_channel: int = 1800
    max_negative_per_channel: int = 3500
    max_iter: int = 140
    routed: bool = True
    models: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)
    classifier_columns: dict = field(default_factory=dict)
    training_info: dict = field(default_factory=dict)

    def fit(self, x, meta, events):
        rng = np.random.default_rng(self.seed)
        blocks, labels, weights = [], [], []
        channel_counts = {}
        for channel in channel_names(x):
            z, eligible = channel_features(x, meta, channel)
            if not eligible.any():
                continue
            y, w = make_channel_targets(meta, events, channel)
            chosen = []
            for fault in SENSOR_TYPES:
                positive = np.flatnonzero(eligible & y[fault].eq(1).to_numpy())
                if len(positive) > self.max_positive_per_type_channel:
                    # Event weights below balance the retained positives. Avoid
                    # weighting both sampling and fitting by inverse duration.
                    positive = rng.choice(positive, self.max_positive_per_type_channel, replace=False)
                chosen.extend(positive)
            negative = np.flatnonzero(eligible & y.eq(0).all(axis=1).to_numpy())
            if len(negative) > self.max_negative_per_channel:
                _, inverse, counts = np.unique(meta.iloc[negative].asset_id.to_numpy(), return_inverse=True, return_counts=True)
                asset_balanced = 1.0/counts[inverse]
                negative = rng.choice(negative, self.max_negative_per_channel, replace=False, p=asset_balanced/asset_balanced.sum())
            ids = np.unique(np.r_[np.asarray(chosen, dtype=int), negative])
            blocks.append(z.iloc[ids]); labels.append(y.iloc[ids]); weights.append(w.iloc[ids])
            channel_counts[channel] = {fault: int((eligible & y[fault].eq(1).to_numpy()).sum()) for fault in SENSOR_TYPES}
        if not blocks:
            raise ValueError("No observed sensor channels available for training")
        inputs = pd.concat(blocks, ignore_index=True)
        target = pd.concat(labels, ignore_index=True)
        event_weights = pd.concat(weights, ignore_index=True)
        self.columns = inputs.columns[inputs.notna().any()].tolist()
        inputs = inputs[self.columns]
        diagnostics = {}
        with threadpool_limits(limits=3):
            for fault in SENSOR_TYPES:
                cols = self.columns
                if self.routed and fault == "SENSOR_NOISE_SPIKES":
                    cols = [name for name in cols if name.startswith(("noise_", "abs_step_", "level_std", "context_change"))
                            or name in ("std_span_ratio", "max_step10_z", "observed_changes60", "running", "steady", "trend_directionality")]
                elif self.routed and fault == "SENSOR_STUCK":
                    cols = [name for name in cols if name.startswith("context_change") or name in
                            ("flat_minutes", "equal_updates", "zero_variance", "flat_fraction60", "observed_changes60", "running", "steady", "noise_ratio60")]
                self.classifier_columns[fault] = cols
                known = target[fault].ge(0)
                y = target.loc[known, fault]
                if y.nunique() < 2:
                    self.models[fault] = float(y.iloc[0]) if len(y) else 0.0
                    diagnostics[fault] = {"positive_samples": int(y.sum()), "negative_samples": int(y.eq(0).sum())}
                    continue
                sample_weights = np.ones(len(y))
                positive = y.eq(1).to_numpy()
                ew = event_weights.loc[known, fault].to_numpy()[positive]
                sample_weights[positive] = ew/ew.mean()*np.sum(~positive)/np.sum(positive)
                monotonic = [1 if fault == "SENSOR_NOISE_SPIKES" and name in ("noise_ratio", "noise_ratio10", "noise_ratio60", "noise_max_ratio10") else 0 for name in cols]
                model = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=.06,
                    max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=3.,
                    monotonic_cst=monotonic, early_stopping=False, random_state=self.seed)
                # Explicit unavailable code avoids histogram corner cases for
                # features absent in the fault-specific subset. It is never a
                # fabricated sensor value or healthy-noise reference.
                model.fit(inputs.loc[known, cols].fillna(-999), y, sample_weight=sample_weights)
                self.models[fault] = model
                diagnostics[fault] = {"positive_samples": int(positive.sum()), "negative_samples": int((~positive).sum()), "features": len(cols)}
        self.training_info = {"examples": len(inputs), "features": len(self.columns),
                              "per_type": diagnostics, "channel_positive_minutes": channel_counts}
        return self

    def predict(self, x, meta):
        """Return per-fault maximum scores across channels observed at that time.

        Outputs are model scores, not calibrated event probabilities. Thresholds
        and persistence must be selected on development data after max pooling.
        """
        scores = pd.DataFrame(0.0, index=x.index, columns=SENSOR_TYPES)
        with threadpool_limits(limits=3):
            for channel in channel_names(x):
                z, eligible = channel_features(x, meta, channel)
                if not eligible.any():
                    continue
                z = z.reindex(columns=self.columns).fillna(-999)
                for fault, model in self.models.items():
                    channel_scores = np.zeros(len(x))
                    if isinstance(model, (float, int)):
                        channel_scores[eligible] = model
                    else:
                        cols = getattr(self, "classifier_columns", {}).get(fault, self.columns)
                        channel_scores[eligible] = model.predict_proba(z.loc[eligible, cols])[:, 1]
                    scores[fault] = np.maximum(scores[fault].to_numpy(), channel_scores)
        return scores.astype(np.float32)
