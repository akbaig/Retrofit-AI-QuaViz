"""Independent review: observation availability, isolation, and raw-prefix replay."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from features import CHANNELS, build_features
from test_causality import toy


PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH" / "RETROFIT_AI_PARTICIPANTS_2026"


def waveform(asset, timestamp, amplitude=0.1, seconds=1.0):
    fs = 10240
    samples = amplitude * np.sin(2 * np.pi * 25 * np.arange(int(fs * seconds)) / fs)
    return dict(asset_id=asset.asset_id, timestamp_utc=timestamp,
                n_samples=len(samples), sample_rate_hz=fs, rpm_at_capture=1500., samples=samples)


def test_snapshot_is_visible_only_after_complete_capture():
    f, asset, context = toy(30)
    start = pd.Timestamp(asset.window_start_utc)
    # The capture begins before minute ten ends, but finishes after that boundary.
    shots = pd.DataFrame([waveform(asset, start + pd.Timedelta(minutes=9, seconds=59.5))])
    meta, x = build_features(f, asset, context, shots)
    assert x.loc[9, "snapshot__rms_g"] != x.loc[9, "snapshot__rms_g"]
    assert x.loc[10, "snapshot__rms_g"] == pytest.approx(0.1 / np.sqrt(2), rel=1e-5)
    assert x.loc[10, "snapshot__age_min"] == pytest.approx(59.5 / 60.)
    for seconds in (599.9, 600.0, 600.49, 600.5, 660.):
        cut = start + pd.Timedelta(seconds=seconds)
        short_meta, short = build_features(f, asset, context, shots, until=cut)
        assert_frame_equal(short, x.loc[meta.available_utc <= cut].reset_index(drop=True))
        assert (short_meta.available_utc <= cut).all()


def test_snapshot_expires_and_does_not_cross_a_stop():
    f, asset, context = toy(390)
    start = pd.Timestamp(asset.window_start_utc)
    shots = pd.DataFrame([waveform(asset, start)])
    _, x = build_features(f, asset, context, shots)
    assert pd.notna(x.loc[359, "snapshot__rms_g"])
    assert pd.isna(x.loc[360, "snapshot__rms_g"])

    f.loc[12:14, "operating_mode"] = "LOCKED_OUT"
    f.loc[12:14, "running_fraction"] = 0.
    shots = pd.DataFrame([waveform(asset, start), waveform(asset, start + pd.Timedelta(minutes=20), .2)])
    _, x = build_features(f, asset, context, shots)
    assert pd.notna(x.loc[11, "snapshot__rms_g"])
    assert x.loc[12:19, "snapshot__rms_g"].isna().all()
    assert x.loc[20, "snapshot__rms_g"] == pytest.approx(.2 / np.sqrt(2), rel=1e-5)


def test_future_mutations_cannot_change_any_past_sensor_or_context_or_waveform_feature():
    f, asset, context = toy(160)
    start = pd.Timestamp(asset.window_start_utc)
    cut = start + pd.Timedelta(minutes=80, seconds=30)
    shots = pd.DataFrame([waveform(asset, start + pd.Timedelta(minutes=10)),
                          waveform(asset, start + pd.Timedelta(minutes=80, seconds=29.9))])
    meta, baseline = build_features(f, asset, context, shots, until=cut)
    changed = f.copy(deep=True)
    future = changed.window_start_utc + pd.Timedelta(minutes=1) > cut
    for channel in CHANNELS:
        for stat in ("mean", "std", "min", "max", "count"):
            changed.loc[future, f"{channel}__{stat}"] = 999999.
    changed.loc[future, "operating_mode"] = "LOCKED_OUT"
    changed.loc[future, "running_fraction"] = 0.
    changed_context = context.copy(deep=True)
    numeric = changed_context.select_dtypes(include=np.number).columns
    changed_context.loc[changed_context.timestamp_utc > cut, numeric] = -999999.
    changed_shots = shots.copy(deep=True)
    changed_shots.at[1, "samples"] = shots.iloc[1].samples * 999999.
    future_meta, future_x = build_features(changed, asset, changed_context, changed_shots)
    assert_frame_equal(baseline, future_x.loc[future_meta.available_utc <= cut].reset_index(drop=True))
    assert (meta.available_utc <= cut).all()


def test_labels_identifiers_and_other_assets_do_not_enter_model_inputs():
    f, asset, context = toy(30)
    start = pd.Timestamp(asset.window_start_utc)
    shots = pd.DataFrame([waveform(asset, start)])
    expected_meta, expected = build_features(f, asset, context, shots)
    banned = ["fault_type", "event_class", "event_id", "event_label", "latent_bearing_wear",
              "detectable_utc", "action_deadline_utc", "scenario_seed", "template_id"]
    for name in banned:
        f[name] = 12345
        context[name] = 12345
        shots[name] = 12345
        asset[name] = 12345
    other_f = f.assign(asset_id="UNRELATED", motor_current_a__mean=1e10)
    other_context = context.assign(asset_id="UNRELATED", duty_demand_pct=1e10)
    other_shots = shots.assign(asset_id="UNRELATED")
    actual_meta, actual = build_features(pd.concat([other_f, f], ignore_index=True), asset,
        pd.concat([other_context, context], ignore_index=True), pd.concat([other_shots, shots], ignore_index=True))
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_meta, expected_meta)
    assert all(np.issubdtype(dtype, np.number) for dtype in actual.dtypes)
    assert not any(name in column for name in banned for column in actual)
    assert not any(name in actual for name in ("asset_id", "timestamp_utc", "window_start_utc"))


@pytest.mark.parametrize("unavailable", [False, True], ids=["zero_observation_count", "absent_channel"])
def test_missing_pressure_evidence_is_missing_in_physics_statistics(unavailable):
    f, asset, context = toy(15)
    channel = "suction_pressure_kpa"
    if unavailable:
        asset.channels_available = "|".join(c for c in CHANNELS if c != channel)
    else:
        f[channel + "__mean"] = np.nan
        f[channel + "__count"] = 0
    # Finite stale aggregate fields must never become physical evidence.
    f[channel + "__std"] = 5.
    f[channel + "__min"] = -50.
    f[channel + "__max"] = 50.
    _, x = build_features(f, asset, context)
    assert x["value__" + channel].isna().all()
    assert x["std__" + channel].isna().all()
    assert x["phys__suction_std_pu"].isna().all()
    assert x["phys__pressure_std_ratio"].isna().all()


@pytest.mark.parametrize("asset_id,day", [("PUMP-01", "2025-08-11"), ("PUMP-04", "2025-06-16")],
                         ids=["real_sparse_training_pump", "real_dense_training_pump"])
def test_training_raw_telemetry_prefix_matches_full_completed_windows(asset_id, day):
    path = SOURCE / "data/full/telemetry" / f"asset_id={asset_id}" / f"day={day}" / "part-0.parquet"
    if not path.exists():
        pytest.skip("Public training telemetry is not present")
    sys.path.insert(0, str(SOURCE / "src"))
    from retrofit_sim.dataset import features_1min

    # Static public ratings are allowed. Only an explicitly named TRAIN telemetry file is read.
    assets = pd.read_csv(SOURCE / "data/full/assets.csv")
    asset = assets.loc[assets.asset_id.eq(asset_id)].iloc[0].copy()
    assert asset.split == "train"
    raw = pd.read_parquet(path)
    raw["asset_id"] = asset_id
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True)
    start = pd.Timestamp(asset.window_start_utc)
    asset.window_end_utc = start + pd.Timedelta(minutes=90)
    raw = raw.loc[raw.timestamp_utc < asset.window_end_utc].copy()
    full_minutes = features_1min(raw)
    meta, full = build_features(full_minutes, asset)
    for minutes, seconds in [(1, 30), (11, 15), (35, 59), (75, 10)]:
        cut = start + pd.Timedelta(minutes=minutes, seconds=seconds)
        prefix_minutes = features_1min(raw.loc[raw.timestamp_utc <= cut])
        short_meta, short = build_features(prefix_minutes, asset, until=cut)
        # The public aggregator sums float32 telemetry; pandas may differ by
        # one float32 ULP when the number of groups changes. This tolerance
        # accommodates that arithmetic noise, not changes to availability.
        assert_frame_equal(short, full.loc[meta.available_utc <= cut].reset_index(drop=True),
                           rtol=2e-5, atol=1e-6)
        assert (short_meta.available_utc <= cut).all()


@pytest.fixture(scope="module")
def fitted_hist_review_inputs():
    """Read observations and an existing model only; no event/label file is read."""
    import joblib
    model_path = Path(__file__).resolve().parents[1] / "runs/hist_01/model.joblib"
    if not model_path.exists():
        pytest.skip("Existing hist_01 model is not present")
    assets = pd.read_csv(SOURCE / "data/full/assets.csv")
    assets = assets.loc[assets.asset_id.isin(["PUMP-08", "PUMP-09"])].copy()
    observations = pd.read_parquet(SOURCE / "data/full/features_1min.parquet",
                                  filters=[("asset_id", "in", assets.asset_id.tolist())])
    observations.window_start_utc = pd.to_datetime(observations.window_start_utc, utc=True)
    context = pd.read_csv(SOURCE / "data/full/operating_context.csv")
    context = context.loc[context.asset_id.isin(assets.asset_id)].copy()
    context.timestamp_utc = pd.to_datetime(context.timestamp_utc, utc=True)
    snapshots = pd.read_parquet(SOURCE / "data/full/vibration_snapshots.parquet",
                               filters=[("asset_id", "in", assets.asset_id.tolist())])
    snapshots.timestamp_utc = pd.to_datetime(snapshots.timestamp_utc, utc=True)
    return joblib.load(model_path), assets, observations, context, snapshots


@pytest.mark.parametrize("asset_id", ["PUMP-08", "PUMP-09"])
def test_fitted_model_first48h_prefix_probabilities_and_alerts(fitted_hist_review_inputs, asset_id):
    from alerts import AlertPolicy, make_alerts
    from infer import infer_asset

    model, assets, observations, contexts, snapshots = fitted_hist_review_inputs
    asset = assets.loc[assets.asset_id.eq(asset_id)].iloc[0].copy()
    start = pd.Timestamp(asset.window_start_utc)
    end = start + pd.Timedelta(hours=48)
    f = observations.loc[observations.asset_id.eq(asset_id) & observations.window_start_utc.lt(end)].copy()
    context = contexts.loc[contexts.asset_id.eq(asset_id) & contexts.timestamp_utc.le(end)].copy()
    shots = snapshots.loc[snapshots.asset_id.eq(asset_id) & snapshots.timestamp_utc.lt(end)].copy()
    meta, x = build_features(f, asset, context, shots, until=end)
    transformed = model.transform(x, meta)
    probability = model.probabilities_from_transformed(transformed)
    # A permissive fixed policy exercises actual emissions and recurrence.
    policy = AlertPolicy(threshold=.02, persistence=3, recovery=60, reminder=1440)
    full_alerts = make_alerts(meta, probability, policy, x=transformed)
    assert len(full_alerts), "The replay must exercise actual alerts"
    cut = start + pd.Timedelta(hours=31, minutes=17, seconds=29)
    # Keep the partial final minute in the supplied aggregates; explicit time
    # must prevent that completed row from being used prematurely.
    short_f = f.loc[f.window_start_utc.le(cut)].copy()
    short_context = context.loc[context.timestamp_utc.le(cut)].copy()
    short_shots = shots.loc[shots.timestamp_utc.le(cut)].copy()
    short_meta, short_x = build_features(short_f, asset, short_context, short_shots, until=cut)
    short_tx = model.transform(short_x, short_meta)
    short_probability = model.probabilities_from_transformed(short_tx)
    eligible = meta.available_utc.le(cut)
    assert_frame_equal(short_tx, transformed.loc[eligible].reset_index(drop=True))
    assert_frame_equal(short_probability, probability.loc[eligible].reset_index(drop=True))
    expected = full_alerts.loc[pd.to_datetime(full_alerts.alert_time_utc, utc=True).le(cut)].reset_index(drop=True)
    actual = infer_asset(model, policy, short_f, asset, short_context, short_shots, until=cut)
    assert_frame_equal(actual, expected)

    # Mutate all future modalities while retaining their rows. The inference
    # boundary must isolate the full fitted model and its incident state.
    numeric = f.select_dtypes(include=np.number).columns
    f = f.astype({name: float for name in numeric})
    f.loc[f.window_start_utc + pd.Timedelta(minutes=1) > cut, numeric] = 999999.
    context_numeric = context.select_dtypes(include=np.number).columns
    context = context.astype({name: float for name in context_numeric})
    context.loc[context.timestamp_utc > cut, context_numeric] = 999999.
    for index in shots.index[shots.timestamp_utc + pd.to_timedelta(shots.n_samples/shots.sample_rate_hz, unit="s") > cut]:
        shots.at[index, "samples"] = np.asarray(shots.loc[index, "samples"]) * 999999.
    mutated = infer_asset(model, policy, f, asset, context, shots, until=cut)
    assert_frame_equal(mutated, actual)


def test_explicit_clock_reveals_terminal_gap_without_future_rows():
    from infer import infer_asset
    from alerts import AlertPolicy
    from learning import FAULTS

    class ClockOnlyModel:
        def transform(self, x, meta):
            return x

        def probabilities_from_transformed(self, x):
            p = pd.DataFrame(0., index=x.index, columns=FAULTS)
            p["COMMUNICATION_LOSS"] = x["obs__gap"]
            return p

    f, asset, _ = toy(180)
    start = pd.Timestamp(asset.window_start_utc)
    f = f.iloc[:60].copy()
    policy = AlertPolicy(threshold=.8, persistence=3)
    implicit = infer_asset(ClockOnlyModel(), policy, f, asset)
    assert implicit.empty
    explicit = infer_asset(ClockOnlyModel(), policy, f, asset, until=start+pd.Timedelta(minutes=80, seconds=15))
    assert len(explicit) == 1
    assert pd.Timestamp(explicit.iloc[0].alert_time_utc) == start+pd.Timedelta(minutes=63)
