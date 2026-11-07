"""Stage 2: causal window features at every decision time (one every ``STRIDE`` minutes).

Per pump the table joins
  * the raw-signal window features and static features (``features.py``);
  * features of the stage-1 deviations ``zA_*`` / ``zB_*``, computed on steady running minutes up to the decision time:
      ``<z>_m30|m120|m360``  mean deviation (size and direction)
      ``<z>_step``           last 30 min against the mean of [t-6 h, t-2 h] (a step, as in a calibration shift)
      ``<z>_max30``          largest |deviation| in the last 30 min (spikes)
      ``<z>_rel``            last 30 min against the pump's own median over [t-24 h, t-2 h] (per-pump offset removed)
    plus cross-channel summaries ``zx_*`` per model: largest and second-largest |m30|, the gap between them (one
    channel deviating alone points to a sensor) and how many channels exceed 2 and 4;
  * training labels (``labels.training_labels``), never used as inputs.

Training pumps get deviations from their out-of-fold stage-1 model (``stage1.model_for``), every other pump from
``final``. Tables are cached per fleet in ``detector/cache/stage2/<fleet>.parquet``.

    python -m quaviz_detector.stage2 build            # official pumps and every simulated fleet, in parallel
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import data
from .features import minute_frame, static_features, window_features
from .labels import training_labels
from .paths import CACHE, SIM_FLEETS
from .stage1 import HOLDOUT_SEEDS, TRAIN_SEEDS, load_model, model_for

STRIDE = 5
STORE = CACHE / "stage2"
DEV_PREFIXES = ("zA_", "zB_", "zx_")
META = ["asset_id", "decision_time", "running", "rows"]
LABELS = ["lab_state", "lab_types", "lab_weights", "lab_events", "lab_benign"]


def deviation_features(z: pd.DataFrame, steady: pd.Series) -> pd.DataFrame:
    zs = z.where(steady, axis=0)
    m30 = zs.rolling(30, min_periods=10).mean()
    parts = {"m30": m30, "m120": zs.rolling(120, min_periods=30).mean(), "m360": zs.rolling(360, min_periods=60).mean(),
             "step": m30 - zs.shift(120).rolling(240, min_periods=60).mean(),
             "max30": zs.abs().rolling(30, min_periods=10).max(),
             "rel": (zs - zs.shift(120).rolling(1320, min_periods=120).median()).rolling(30, min_periods=10).mean()}
    out = pd.concat({k: v for k, v in parts.items()}, axis=1)
    out.columns = [f"{c}_{k}" for k, c in out.columns]
    for model in ("A", "B"):
        a = m30.filter(like=f"z{model}_").abs().to_numpy()
        with np.errstate(invalid="ignore"):
            srt = -np.sort(-np.nan_to_num(a, nan=-np.inf), axis=1)
        top = np.where(np.isfinite(srt[:, 0]), srt[:, 0], np.nan)
        second = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan) if a.shape[1] > 1 else np.full(len(a), np.nan)
        out[f"zx_{model}_top"] = top
        out[f"zx_{model}_second"] = second
        out[f"zx_{model}_gap"] = top - second
        out[f"zx_{model}_n2"] = np.where(np.isfinite(top), (a > 2).sum(axis=1), np.nan)
        out[f"zx_{model}_n4"] = np.where(np.isfinite(top), (a > 4).sum(axis=1), np.nan)
    return out


def pump_table(ds: data.Dataset, aid: str, models: dict) -> pd.DataFrame:
    fr = minute_frame(ds, aid)
    tag = model_for(aid)
    if tag not in models:
        models[tag] = load_model(tag)
    z = models[tag].deviations(fr, ds.asset(aid))
    tab = pd.concat([window_features(fr), deviation_features(z, fr["steady"])], axis=1).iloc[STRIDE - 1::STRIDE]
    for k, v in static_features(ds.asset(aid)).items():
        tab[k] = v
    lab = training_labels(tab.index, ds, aid, tab["running"].to_numpy(bool))
    num = [c for c in tab.columns if c not in META and tab[c].dtype != object]
    tab[num] = tab[num].astype(np.float32)
    return pd.concat([tab, lab], axis=1)


def fleets() -> list[tuple[str, str]]:
    return [("official", "")] + [(f"s{s:04d}", str(SIM_FLEETS / f"s{s:04d}")) for s in TRAIN_SEEDS + HOLDOUT_SEEDS]


def build_one(name: str, path: str) -> str:
    t0 = time.time()
    ds = data.load() if name == "official" else data.load(path, name=name)
    models: dict = {}
    tab = pd.concat([pump_table(ds, aid, models) for aid in ds.ids()])
    tab.index.name = "window_start_utc"
    STORE.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(STORE / f"{name}.parquet")
    return f"{name}: {ds.assets.shape[0]} pumps, {len(tab)} rows, {tab.shape[1]} columns, {time.time() - t0:.0f} s"


def cache_name(asset_id: str) -> str:
    return f"s{asset_id[3:7]}" if asset_id.startswith("SIM") else "official"


def load(asset_ids) -> pd.DataFrame:
    """Stage-2 tables of the given pumps, from the cache."""
    ids = list(asset_ids)
    by = {}
    for a in ids:
        by.setdefault(cache_name(a), []).append(a)
    parts = [pd.read_parquet(STORE / f"{n}.parquet", filters=[("asset_id", "in", v)]) for n, v in by.items()]
    return pd.concat(parts) if parts else pd.DataFrame()


def build(jobs: int = 6, only=None) -> None:
    from joblib import Parallel, delayed
    todo = [f for f in fleets() if only is None or f[0] in only]
    for msg in Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(build_one)(n, p) for n, p in todo):
        print(msg, flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m quaviz_detector.stage2")
    p.add_argument("cmd", choices=["build"])
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--only", nargs="*", default=None, help="Fleet names, e.g. official s0101.")
    a = p.parse_args(argv)
    build(a.jobs, a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
