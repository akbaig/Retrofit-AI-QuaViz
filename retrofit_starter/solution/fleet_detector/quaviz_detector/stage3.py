"""Stage 3 orchestration: classifier fits, cross-fitted probabilities and alert-policy tuning.

A configuration fixes the training sources (simulated training seeds, official train pumps, or both) and whether the
stage-1 deviation features are used. The alert policy is tuned on out-of-fold probabilities of the training pumps
(the stage-1 folds, so no pump is scored by a classifier that saw it), on the official scorer, maximizing the mean of
the composite on the simulated pumps and on the official train pumps. Validation, the held-out seeds and the test
pumps play no part in it.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass

import pandas as pd

from . import data, stage2
from .classifier import FaultClassifier
from .paths import CACHE, SIM_FLEETS
from .policy import Policy, grid, smooth
from .scorer import score_assets
from .stage1 import FOLDS, HOLDOUT_SEEDS, OFFICIAL_TRAIN, TRAIN_SEEDS, fold_index

STORE = CACHE / "stage3"


@dataclass(frozen=True)
class Config:
    name: str = "quaviz"
    sim: bool = True
    official: bool = True
    deviations: bool = True


CONFIGS = {c.name: c for c in [Config(), Config("quaviz_nodev", deviations=False), Config("quaviz_simonly", official=False),
                               Config("quaviz_officialonly", sim=False)]}


def sim_ids(seeds) -> list[str]:
    out = []
    for s in seeds:
        out += sorted(pd.read_parquet(stage2.STORE / f"s{s:04d}.parquet", columns=["asset_id"])["asset_id"].unique())
    return out


def training_ids(cfg: Config, official_ids, seeds=TRAIN_SEEDS) -> list[str]:
    return (sorted(official_ids) if cfg.official else []) + (sim_ids(seeds) if cfg.sim else [])


def _key(cfg: Config, ids) -> str:
    return f"{cfg.name}_{hashlib.sha1('|'.join(sorted(ids)).encode()).hexdigest()[:10]}"


def fit_classifier(cfg: Config, ids, log=print) -> FaultClassifier:
    """Fit (or load from cache) the classifier of ``cfg`` on the given training pumps."""
    path = STORE / f"{_key(cfg, ids)}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    clf = FaultClassifier(deviations=cfg.deviations).fit(stage2.load(ids))
    STORE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)
    log(f"[{cfg.name}] classifier on {len(ids)} pumps, {clf.n_rows} rows, {len(clf.cols)} features: {time.time() - t0:.0f} s")
    return clf


def predict_proba(clf: FaultClassifier, ids) -> pd.DataFrame:
    return clf.predict_proba(stage2.load(ids))


def oof_probs(cfg: Config, log=print) -> pd.DataFrame:
    """Out-of-fold class probabilities of every training pump of ``cfg`` (official train and simulated seeds)."""
    path = STORE / f"{cfg.name}_oof.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ids = training_ids(cfg, OFFICIAL_TRAIN)
    parts = []
    for j in range(len(FOLDS)):
        held = [a for a in ids if fold_index(a) == j]
        clf = fit_classifier(cfg, [a for a in ids if fold_index(a) != j], log)
        parts.append(predict_proba(clf, held))
    out = pd.concat(parts)
    out.to_parquet(path)
    return out


_LABELS: dict = {}


def labels(kind: str) -> data.Dataset:
    """Label tables for scoring: ``official`` or ``sim`` (all simulated fleets pooled)."""
    if kind not in _LABELS:
        if kind == "official":
            _LABELS[kind] = data.load_labels()
        else:
            _LABELS[kind] = data.concat([data.load_labels(SIM_FLEETS / f"s{s:04d}", f"s{s:04d}") for s in TRAIN_SEEDS + HOLDOUT_SEEDS],
                                        name="sim")
    return _LABELS[kind]


def _score_chunk(policies, sets):
    rows, cache = [], {}
    for p in policies:
        r = {"policy": p}
        for name, (probs, ds, ids) in sets.items():
            if (name, p.m) not in cache:
                cache[(name, p.m)] = smooth(probs, p.m)
            s = score_assets(p.alerts(probs, cache[(name, p.m)]), ds, ids, name)
            r |= {f"{name}_composite": s["composite_score"], f"{name}_f1": s["event_f1"],
                  f"{name}_type_recall": s["fault_type_recall"], f"{name}_fa_per_day": s["false_alerts_per_asset_day"]}
        rows.append(r)
    return rows


def tune(cfg: Config, official_ids=OFFICIAL_TRAIN, jobs: int = 8, log=print) -> tuple[Policy, pd.DataFrame]:
    """Best policy on the out-of-fold probabilities of the training pumps; ``official_ids`` limits which official
    train pumps take part (leave-one-pump-out drops the held-out pump)."""
    from joblib import Parallel, delayed
    off = sorted(set(official_ids) & set(OFFICIAL_TRAIN)) if cfg.official else []
    cand = grid()
    path = STORE / f"{_key(cfg, off + [repr(c) for c in cand])}_policy.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    oof = oof_probs(cfg, log)
    t0 = time.time()
    sets = {}
    if cfg.sim:
        ids = sorted(a for a in oof["asset_id"].unique() if a.startswith("SIM"))
        sets["sim"] = (oof[oof["asset_id"].isin(ids)], labels("sim"), ids)
    if off:
        sets["official"] = (oof[oof["asset_id"].isin(off)], labels("official"), off)
    chunks = [cand[i::jobs] for i in range(jobs)]
    rows = [r for part in Parallel(n_jobs=jobs)(delayed(_score_chunk)(c, sets) for c in chunks) for r in part]
    res = pd.DataFrame(rows)
    res["objective"] = res[[f"{k}_composite" for k in sets]].mean(axis=1)
    res = res.sort_values("objective", ascending=False).reset_index(drop=True)
    with open(path, "wb") as fh:
        pickle.dump((res["policy"].iloc[0], res), fh)
    log(f"[{cfg.name}] policy tuned on {', '.join(sets)} ({len(cand)} candidates): {time.time() - t0:.0f} s")
    return res["policy"].iloc[0], res
