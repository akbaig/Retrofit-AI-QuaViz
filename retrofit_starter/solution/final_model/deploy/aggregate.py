"""Turn buffered 1 s rows into the challenge's public 1-min feature schema."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from paths import SOURCE

_SRC = str(SOURCE / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from retrofit_sim.dataset import FEATURE_CHANNELS, features_1min  # noqa: E402
from retrofit_sim.generate import ALL_TELEMETRY_COLUMNS  # noqa: E402

REQUIRED = ("timestamp_utc", "operating_mode")


def normalize_row(payload: dict, asset_id: str | None = None) -> dict:
    """Map a replay/MQTT payload to a telemetry row."""
    row = dict(payload)
    aid = asset_id or row.get("asset_id")
    if not aid:
        raise ValueError("asset_id missing from payload and topic")
    row["asset_id"] = aid
    ts = pd.Timestamp(row["timestamp_utc"])
    row["timestamp_utc"] = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if "quality_flags" not in row or row["quality_flags"] is None:
        row["quality_flags"] = 0
    row["quality_flags"] = int(row["quality_flags"])
    if "operating_mode" not in row or row["operating_mode"] is None:
        row["operating_mode"] = "UNKNOWN"
    for c in ALL_TELEMETRY_COLUMNS:
        if c not in row:
            row[c] = float("nan")
    return row


def rows_to_features(rows: list[dict]) -> pd.DataFrame:
    """Aggregate one or more completed minutes with the official features_1min()."""
    if not rows:
        return pd.DataFrame()
    tele = pd.DataFrame(rows)
    tele["timestamp_utc"] = pd.to_datetime(tele["timestamp_utc"], utc=True)
    # Ensure columns exist for the official aggregator.
    for c in FEATURE_CHANNELS:
        if c not in tele.columns:
            tele[c] = float("nan")
    if "quality_flags" not in tele.columns:
        tele["quality_flags"] = 0
    return features_1min(tele)


def asset_id_from_topic(topic: str | None) -> str | None:
    """retrofit/<site>/<asset>/telemetry → asset id."""
    if not topic:
        return None
    parts = topic.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "retrofit":
        return parts[2]
    return None
