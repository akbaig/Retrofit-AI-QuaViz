"""Thin wrapper around the official scorer (``retrofit_sim.scoring.score``), which is used unchanged."""
from __future__ import annotations

import pandas as pd
from retrofit_sim.scoring import score

from .data import Dataset

PRED_COLS = ["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"]


def score_assets(preds: pd.DataFrame, ds: Dataset, asset_ids, label: str = "eval") -> dict:
    """Score predictions on a set of assets, pooled as one split named ``label``."""
    ids = set(asset_ids)
    assets = ds.assets[ds.assets["asset_id"].isin(ids)].copy()
    assets["split"] = label
    events = ds.events[ds.events["asset_id"].isin(ids)]
    benign = ds.benign[ds.benign["asset_id"].isin(ids)] if len(ds.benign) else None
    preds = preds if len(preds) else pd.DataFrame(columns=PRED_COLS)
    return score(preds, events, assets, benign, label)
