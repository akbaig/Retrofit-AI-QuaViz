"""Stage-2 rows (window features plus stage-1 deviation features) must not change when later data is removed."""
import numpy as np
import pandas as pd
import pytest

from quaviz_detector import data, stage1, stage2


@pytest.fixture(scope="module")
def official():
    if not (stage1.STORE / "final.pkl").exists():
        pytest.skip("stage-1 models not fitted (python -m quaviz_detector.stage1 fit)")
    return data.load()


def _truncate(ds, cut):
    t = ds.subset(ds.ids())
    t.features = t.features[t.features["window_start_utc"] + pd.Timedelta(minutes=1) <= cut]
    t.context = t.context[t.context["timestamp_utc"] < cut]
    return t


@pytest.mark.parametrize("asset_id", ["PUMP-02", "PUMP-08", "PUMP-11"])
def test_stage2_rows_are_causal(official, asset_id):
    ds = official.subset([asset_id])
    models: dict = {}
    full = stage2.pump_table(ds, asset_id, models)
    cols = [c for c in full.drop(columns=stage2.LABELS).select_dtypes(["number", "bool"]).columns]
    start = full["decision_time"].iloc[0]
    for h in sorted(np.random.default_rng(1).uniform(12, 160, 3)):
        cut = start + pd.Timedelta(hours=float(h))
        part = stage2.pump_table(_truncate(ds, cut), asset_id, models)
        a = full[full["decision_time"] <= cut][cols].to_numpy(float)
        b = part[part["decision_time"] <= cut][cols].to_numpy(float)
        assert a.shape == b.shape
        assert np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-6), f"{asset_id}: stage-2 rows before {cut} changed"
