"""Features at a decision time must not change when later data is removed."""
import numpy as np
import pandas as pd
import pytest

from quaviz_detector import data
from quaviz_detector.features import asset_features


@pytest.fixture(scope="module")
def official():
    return data.load()


def _truncate(ds, cut):
    t = ds.subset(ds.ids())
    t.features = t.features[t.features["window_start_utc"] + pd.Timedelta(minutes=1) <= cut]
    t.context = t.context[t.context["timestamp_utc"] < cut]
    return t


@pytest.mark.parametrize("asset_id", ["PUMP-01", "PUMP-04", "PUMP-05"])
def test_features_are_causal(official, asset_id):
    ds = official.subset([asset_id])
    full = asset_features(ds, asset_id)
    start = full["decision_time"].iloc[0]
    rng = np.random.default_rng(0)
    for h in sorted(rng.uniform(6, 160, 3)):
        cut = start + pd.Timedelta(hours=float(h))
        part = asset_features(_truncate(ds, cut), asset_id)
        keep = full["decision_time"] <= cut
        a = full[keep].select_dtypes("number").to_numpy(float)
        b = part[part["decision_time"] <= cut][full.select_dtypes("number").columns].to_numpy(float)
        assert a.shape == b.shape
        assert np.allclose(a, b, equal_nan=True), f"{asset_id}: features before {cut} changed after truncation"
