"""Sensor target specificity, causal history and anonymous channel transfer."""
import numpy as np
import pandas as pd

from sensor_expert import SensorExpert, SENSOR_TYPES, channel_features, make_channel_targets


def toy_sensor(channel="discharge_pressure_kpa", n=420, multiplier=1.0):
    rng = np.random.default_rng(71)
    noise = np.where(np.arange(n) >= 240, 5.0, 1.0)
    level = multiplier*(1.0+rng.normal(size=n)*.001*noise)
    std = multiplier*.01*noise
    t = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    meta = pd.DataFrame({"asset_id": "A", "available_utc": t})
    x = pd.DataFrame({"level__"+channel: level, "std__"+channel: std,
        "range__"+channel: 5*std, "fresh__"+channel: np.ones(n),
        "age__"+channel: np.zeros(n), "count__"+channel: np.ones(n),
        "flat_minutes__"+channel: np.zeros(n), "equal_updates__"+channel: np.ones(n),
        "obs__running": np.ones(n), "obs__steady": np.ones(n), "obs__gap": np.zeros(n)})
    event = pd.DataFrame([dict(asset_id="A", fault_type="SENSOR_NOISE_SPIKES",
        sensor_tag=channel, onset_utc=t[240], detectable_utc=t[245], end_utc=t[-1], observable=True)])
    return x, meta, event


def test_target_is_sensor_specific_and_previsible_unknown():
    x, meta, event = toy_sensor()
    y, weights = make_channel_targets(meta, event, "discharge_pressure_kpa")
    assert y.loc[239, "SENSOR_NOISE_SPIKES"] == 0
    assert (y.loc[240:244, "SENSOR_NOISE_SPIKES"] == -1).all()
    assert y.loc[245, "SENSOR_NOISE_SPIKES"] == 1
    other, _ = make_channel_targets(meta, event, "motor_current_a")
    assert other.eq(0).all().all()
    event.loc[0, "observable"] = False
    unknown, _ = make_channel_targets(meta, event, "discharge_pressure_kpa")
    assert unknown.loc[245, "SENSOR_NOISE_SPIKES"] == -1
    assert weights.loc[245, "SENSOR_NOISE_SPIKES"] > 0


def test_feature_prefix_causality_and_no_identity_columns():
    x, meta, event = toy_sensor()
    full, mask = channel_features(x, meta, "discharge_pressure_kpa")
    for stop in (1, 80, 200, 280):
        prefix, prefix_mask = channel_features(x.iloc[:stop], meta.iloc[:stop], "discharge_pressure_kpa")
        pd.testing.assert_frame_equal(prefix, full.iloc[:stop])
        np.testing.assert_array_equal(prefix_mask, mask[:stop])
    assert not any("discharge" in c or "asset" in c or "time" in c or "rating" in c for c in full.columns)
    np.testing.assert_allclose(full.loc[200, "noise_ratio60"], 1.)
    np.testing.assert_allclose(full.loc[320, "noise_ratio60"], 5.)


def test_channel_transfer_is_scale_invariant():
    source, meta, _ = toy_sensor("discharge_pressure_kpa")
    target, _, _ = toy_sensor("motor_current_a", multiplier=100)
    a, amask = channel_features(source, meta, "discharge_pressure_kpa")
    b, bmask = channel_features(target, meta, "motor_current_a")
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(amask, bmask)


def test_history_does_not_cross_asset_boundaries():
    x, meta, _ = toy_sensor(n=420)
    combined_x = pd.concat([x, x*1.0], ignore_index=True)
    second = meta.assign(asset_id="B")
    combined_meta = pd.concat([meta, second], ignore_index=True)
    full, _ = channel_features(combined_x, combined_meta, "discharge_pressure_kpa")
    pd.testing.assert_frame_equal(full.iloc[:420].reset_index(drop=True), full.iloc[420:].reset_index(drop=True))


def test_shared_classifier_transfers_to_an_unseen_tag():
    x, meta, events = toy_sensor()
    expert = SensorExpert(max_iter=45).fit(x, meta, events)
    source = expert.predict(x, meta)
    target_x, target_meta, _ = toy_sensor("motor_current_a", multiplier=100)
    target = expert.predict(target_x, target_meta)
    np.testing.assert_allclose(source.to_numpy(), target.to_numpy(), atol=1e-6)
    assert source.loc[320:, "SENSOR_NOISE_SPIKES"].median() > .8
    assert source.loc[120:220, "SENSOR_NOISE_SPIKES"].median() < .2
    assert list(source.columns) == list(SENSOR_TYPES)


def test_missing_channel_is_not_sensor_fault_evidence():
    x, meta, events = toy_sensor()
    expert = SensorExpert(max_iter=10).fit(x, meta, events)
    x.loc[:, "level__discharge_pressure_kpa"] = np.nan
    x.loc[:, "fresh__discharge_pressure_kpa"] = 0
    out = expert.predict(x, meta)
    assert out.eq(0).all().all()
