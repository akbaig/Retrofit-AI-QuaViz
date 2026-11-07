"""One loader for the official challenge data and for fleets we generate with the challenge simulator.

Both layouts are supported:
  * official: a flat folder (``data/full``) with assets.csv, features_1min.parquet, events_*.csv, ...
  * generated: ``<dir>/public`` with the same tables plus ``<dir>/private`` with full labels and true states.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from .paths import OFFICIAL_DATA


def ts(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601")


@dataclass
class Dataset:
    name: str
    assets: pd.DataFrame
    features: pd.DataFrame          # 1-min aggregates per channel (window_start_utc is tz-aware)
    context: pd.DataFrame           # 1-min operating context (timestamp_utc is tz-aware)
    events: pd.DataFrame            # all labels we are allowed to see, timestamps kept as ISO strings for the scorer
    benign: pd.DataFrame
    alarms: pd.DataFrame
    maintenance: pd.DataFrame
    truth: pd.DataFrame | None = None   # true plant and latent states (generated fleets only)

    def ids(self, split: str | None = None) -> list[str]:
        a = self.assets if split is None else self.assets[self.assets["split"] == split]
        return sorted(a["asset_id"])

    def subset(self, ids) -> "Dataset":
        keep = set(ids)
        parts = {}
        for f in fields(self):
            v = getattr(self, f.name)
            parts[f.name] = v[v["asset_id"].isin(keep)].reset_index(drop=True) if isinstance(v, pd.DataFrame) else v
        return Dataset(**parts)

    def asset(self, asset_id: str) -> pd.Series:
        return self.assets.set_index("asset_id").loc[asset_id]


def _csvs(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["asset_id"])


def load(path=OFFICIAL_DATA, name: str | None = None) -> Dataset:
    p = Path(path)
    pub = p / "public" if (p / "public").is_dir() else p
    priv = p / "private" if (p / "private").is_dir() else None
    assets = pd.read_csv(pub / "assets.csv")
    feats = pd.read_parquet(pub / "features_1min.parquet")
    feats["window_start_utc"] = ts(feats["window_start_utc"])
    ctx = pd.read_csv(pub / "operating_context.csv")
    ctx["timestamp_utc"] = ts(ctx["timestamp_utc"])
    ev_paths = sorted(pub.glob("events_*.csv")) + (sorted(priv.glob("events_*.csv")) if priv else [])
    events = _csvs(ev_paths)
    if len(events):
        events = events.drop_duplicates("event_id", keep="last").reset_index(drop=True)
    benign = _csvs(sorted(pub.glob("benign_windows_*.csv")) + (sorted(priv.glob("benign_windows_*.csv")) if priv else []))
    alarms = _csvs([pub / "alarms.csv"])
    maint = _csvs([pub / "maintenance.csv"])
    truth = None
    if priv and (priv / "truth_1min.parquet").exists():
        truth = pd.read_parquet(priv / "truth_1min.parquet")
        truth["timestamp_utc"] = ts(truth["timestamp_utc"])
    return Dataset(name or p.name, assets, feats, ctx, events, benign, alarms, maint, truth)


def load_labels(path=OFFICIAL_DATA, name: str | None = None) -> Dataset:
    """Assets, events and benign windows only (enough for scoring), without the telemetry tables."""
    p = Path(path)
    pub = p / "public" if (p / "public").is_dir() else p
    priv = p / "private" if (p / "private").is_dir() else None
    events = _csvs(sorted(pub.glob("events_*.csv")) + (sorted(priv.glob("events_*.csv")) if priv else []))
    if len(events):
        events = events.drop_duplicates("event_id", keep="last").reset_index(drop=True)
    benign = _csvs(sorted(pub.glob("benign_windows_*.csv")) + (sorted(priv.glob("benign_windows_*.csv")) if priv else []))
    empty = pd.DataFrame(columns=["asset_id"])
    return Dataset(name or p.name, pd.read_csv(pub / "assets.csv"), empty, empty, events, benign, empty, empty)


def concat(datasets: list[Dataset], name: str = "combined") -> Dataset:
    parts = {"name": name}
    for f in fields(Dataset):
        if f.name == "name":
            continue
        frames = [getattr(d, f.name) for d in datasets if isinstance(getattr(d, f.name), pd.DataFrame)]
        frames = [x for x in frames if len(x)]
        parts[f.name] = pd.concat(frames, ignore_index=True) if frames else (None if f.name == "truth" else pd.DataFrame(columns=["asset_id"]))
    ids = parts["assets"]["asset_id"]
    if ids.duplicated().any():
        raise ValueError(f"duplicate asset ids across datasets: {sorted(ids[ids.duplicated()])}")
    return Dataset(**parts)
