"""Deploy aggregator uses the official 1-min feature builder."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINAL))
sys.path.insert(0, str(FINAL.parents[1] / "src"))

from deploy.aggregate import normalize_row, rows_to_features  # noqa: E402
from retrofit_sim.dataset import features_1min  # noqa: E402


def test_normalize_and_aggregate_one_minute():
    rows = []
    t0 = pd.Timestamp("2025-09-10T09:00:00Z")
    for i in range(60):
        rows.append(normalize_row({
            "timestamp_utc": (t0 + pd.Timedelta(seconds=i)).isoformat(),
            "operating_mode": "RUNNING",
            "motor_speed_rpm": 1000.0 + i * 0.01,
            "motor_current_a": 40.0,
            "quality_flags": 0,
        }, "PUMP-08"))
    feats = rows_to_features(rows)
    assert len(feats) == 1
    assert feats.iloc[0]["rows"] == 60
    assert feats.iloc[0]["asset_id"] == "PUMP-08"
    # Same path as dataset writer
    tele = pd.DataFrame(rows)
    ref = features_1min(tele)
    pd.testing.assert_series_equal(
        feats.iloc[0][["rows", "running_fraction"]],
        ref.iloc[0][["rows", "running_fraction"]],
        check_names=False,
    )
