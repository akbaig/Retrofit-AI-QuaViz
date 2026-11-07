"""Frozen best driven by a growing 1-min feature table (online / near-real-time)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from alerts import COLS, AlertPolicy
from infer import infer_asset


class OnlineBest:
    """Hold static asset/context/snapshots; re-infer as minutes complete."""

    def __init__(
        self,
        *,
        model_path: Path,
        policy_path: Path,
        data_dir: Path,
        asset_id: str,
    ):
        self.asset_id = asset_id
        self.data_dir = Path(data_dir)
        self.model = joblib.load(model_path)
        self.policy = AlertPolicy(**json.loads(Path(policy_path).read_text()))
        assets = pd.read_csv(self.data_dir / "assets.csv")
        row = assets[assets.asset_id == asset_id]
        if row.empty:
            raise ValueError(f"Unknown asset_id {asset_id}")
        self.asset = row.iloc[0]
        ctx_path = self.data_dir / "operating_context.csv"
        snap_path = self.data_dir / "vibration_snapshots.parquet"
        self.context = pd.read_csv(ctx_path) if ctx_path.exists() else None
        self.snapshots = (
            pd.read_parquet(snap_path, filters=[("asset_id", "==", asset_id)])
            if snap_path.exists()
            else None
        )
        self.features = pd.DataFrame()
        self._seen_alert_keys: set[tuple] = set()

    def extend_features(self, minute_frame: pd.DataFrame) -> None:
        if minute_frame is None or not len(minute_frame):
            return
        self.features = (
            minute_frame.copy()
            if not len(self.features)
            else pd.concat([self.features, minute_frame], ignore_index=True)
        )

    def infer_new_alerts(self, until: pd.Timestamp | None = None) -> pd.DataFrame:
        if not len(self.features):
            return pd.DataFrame(columns=COLS)
        starts = pd.to_datetime(self.features["window_start_utc"], utc=True)
        if until is None:
            # A completed minute [t, t+1) becomes available at t+1.
            until = starts.max() + pd.Timedelta(minutes=1)
        # Do not treat pre-stream history as gateway loss: clip the asset window
        # to the first minute we have actually aggregated from the live feed.
        asset = self.asset.copy()
        first = starts.min()
        if pd.Timestamp(asset.window_start_utc) < first:
            asset["window_start_utc"] = first.isoformat()
        alerts = infer_asset(
            self.model, self.policy, self.features, asset,
            self.context, self.snapshots, until=until,
        )
        if alerts.empty:
            return alerts
        fresh = []
        for _, row in alerts.iterrows():
            key = (row.asset_id, row.alert_time_utc, row.fault_type)
            if key in self._seen_alert_keys:
                continue
            self._seen_alert_keys.add(key)
            fresh.append(row)
        return pd.DataFrame(fresh) if fresh else pd.DataFrame(columns=COLS)
