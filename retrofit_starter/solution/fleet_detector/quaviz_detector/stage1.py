"""Fit and check the stage-1 normal-behavior models.

Training data: healthy minutes of the official train pumps (PUMP-01..07, every minute, extra weight) and of the
simulated training seeds (every 6th minute). Every training pump also needs deviations from a model that did not see
it, so pumps are split into folds (simulated pumps by seed, official train pumps spread over the folds) and each fold
is scored by a model fitted on the other folds. ``final`` is fitted on everything; it scores the validation and test
pumps and the held-out simulated seeds, none of which it saw.

    python -m quaviz_detector.stage1 fit                 # writes detector/cache/stage1/*.pkl
    python -m quaviz_detector.stage1 report              # separability vs B1's z-scores on held-out pumps
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from . import data
from .features import minute_frame
from .labels import asset_events, in_any_event
from .normal import CONTEXT_INPUTS, MEASURED_INPUTS, TARGETS_A, NormalModel
from .paths import CACHE, RESULTS, SIM_FLEETS

TRAIN_SEEDS = list(range(101, 112))
HOLDOUT_SEEDS = [112, 113]
FOLDS = [[101, 104, 107, 110], [102, 105, 108, 111], [103, 106, 109]]
OFFICIAL_TRAIN = [f"PUMP-{k:02d}" for k in range(1, 8)]
OFFICIAL_FOLD = {aid: j % len(FOLDS) for j, aid in enumerate(OFFICIAL_TRAIN)}
OFFICIAL_WEIGHT = 4.0
SIM_EVERY = 6
KEEP = sorted(set(TARGETS_A + MEASURED_INPUTS + CONTEXT_INPUTS + ["run_age_min", "schedule", "running", "steady"]))
STORE = CACHE / "stage1"


def fleet(seed: int) -> data.Dataset:
    return data.load(SIM_FLEETS / f"s{seed:04d}", name=f"s{seed:04d}")


def healthy_samples(ds: data.Dataset, ids=None, every: int = 1, weight: float = 1.0) -> dict:
    out = {}
    for aid in ids or ds.ids():
        fr = minute_frame(ds, aid)
        mask = fr["steady"].to_numpy(bool) & ~in_any_event(fr.index, asset_events(ds, aid))
        out[aid] = (fr[[c for c in KEEP if c in fr]], ds.asset(aid), mask, every, weight)
    return out


def fold_index(asset_id: str) -> int | None:
    """Cross-fit fold of a training pump; None for pumps no stage-1 model is trained on."""
    if asset_id in OFFICIAL_FOLD:
        return OFFICIAL_FOLD[asset_id]
    if asset_id.startswith("SIM"):
        seed = int(asset_id[3:7])
        return next((j for j, f in enumerate(FOLDS) if seed in f), None)
    return None


def model_for(asset_id: str) -> str:
    """Which stage-1 model scores an asset: its out-of-fold model for training pumps, else ``final``."""
    j = fold_index(asset_id)
    return "final" if j is None else f"fold{j}"


def load_model(tag: str) -> NormalModel:
    with open(STORE / f"{tag}.pkl", "rb") as fh:
        return pickle.load(fh)


def fit(max_iter: int = 300) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    samples = healthy_samples(data.load(), OFFICIAL_TRAIN, every=1, weight=OFFICIAL_WEIGHT)
    n_off = sum(int(s[2].sum()) for s in samples.values())
    for s in TRAIN_SEEDS:
        samples |= healthy_samples(fleet(s), every=SIM_EVERY)
    print(f"loaded {len(samples)} training pumps ({n_off} official healthy minutes) in {time.time() - t0:.0f} s", flush=True)
    jobs = [(f"fold{j}", j) for j in range(len(FOLDS))] + [("final", None)]
    for tag, held in jobs:
        t1 = time.time()
        use = [v for aid, v in samples.items() if held is None or fold_index(aid) != held]
        m = NormalModel(max_iter=max_iter).fit(use)
        with open(STORE / f"{tag}.pkl", "wb") as fh:
            pickle.dump(m, fh)
        print(f"{tag}: {len(use)} pumps, {len(m.mean)} channel models, {time.time() - t1:.0f} s", flush=True)


# ---- report -------------------------------------------------------------------------------------------------------
def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _score(z: pd.DataFrame, steady: pd.Series) -> pd.Series:
    """Anomaly score per minute: largest 30-min mean |deviation| over channels (steady minutes only)."""
    r = z.where(steady, axis=0).abs().rolling(30, min_periods=10).mean()
    return r.max(axis=1, skipna=True)


def _relative(z: pd.DataFrame, steady: pd.Series) -> pd.DataFrame:
    """Deviation relative to the pump's own recent past: minus the median over [t-24 h, t-2 h] (past data only)."""
    zs = z.where(steady, axis=0)
    return zs - zs.shift(120).rolling(1320, min_periods=120).median()


def report(out=None) -> pd.DataFrame:
    from retrofit_sim import baselines as B
    models = {}
    model = lambda aid: models.setdefault(model_for(aid), load_model(model_for(aid)))
    official = data.load()
    sets = [("official", official, official.ids("train") + official.ids("validation"))] + \
           [(f"sim s{s}", fleet(s), None) for s in HOLDOUT_SEEDS]
    rows, healthy, pooled = [], [], {}
    rng = np.random.default_rng(0)
    for label, ds, ids in sets:
        for aid in ids or ds.ids():
            a = ds.asset(aid)
            fr = minute_frame(ds, aid)
            ev = asset_events(ds, aid)
            z = model(aid).deviations(fr, a)
            sub = ds.subset([aid])
            zb1 = B.zscores(B.regime_frame(sub.features, sub.assets)).set_index("window_start_utc").reindex(fr.index)
            zb1 = zb1[[c for c in zb1.columns if c.startswith("z_")]].astype(float)
            st = fr["steady"]
            scores = {"B1 z (first 24 h)": _score(zb1, st), "ours A (context)": _score(z.filter(like="zA_"), st),
                      "ours B (consistency)": _score(z.filter(like="zB_"), st),
                      "ours A+B relative to own past": _score(_relative(z, st), st)}
            hmask = st.to_numpy(bool) & ~in_any_event(fr.index, ev)
            for k, v in z.items():
                vv = v.to_numpy()[hmask]
                vv = vv[np.isfinite(vv)]
                if len(vv):
                    healthy.append({"set": label, "asset_id": aid, "deviation": k, "median_abs": np.median(np.abs(vv)),
                                    "frac_gt3": float((np.abs(vv) > 3).mean())})
            for name, s in scores.items():
                h = s.to_numpy()[hmask]
                pooled.setdefault((label, name), []).append(rng.choice(h, min(len(h), 3000), replace=False) if len(h) else h)
            for _, e in ev[ev["observable"]].iterrows():
                win = (fr.index >= e["detectable_utc"]) & (fr.index <= min(e["end_eff"], e["detectable_utc"] + pd.Timedelta(hours=6)))
                for name, s in scores.items():
                    rows.append({"set": label, "asset_id": aid, "event_id": e["event_id"], "fault_type": e["fault_type"],
                                 "score": name, "values": s.to_numpy()[win]})
    neg = {k: np.concatenate(v) for k, v in pooled.items()}
    for r in rows:
        r["auc"] = _auc(r.pop("values"), neg[(r["set"], r["score"])])
    res = pd.DataFrame(rows)
    hz = pd.DataFrame(healthy)
    out = out or RESULTS / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "separability_per_event.csv", index=False)
    hz.to_csv(out / "healthy_deviation.csv", index=False)
    return res, hz


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m quaviz_detector.stage1")
    p.add_argument("cmd", choices=["fit", "report"])
    p.add_argument("--max-iter", type=int, default=300)
    a = p.parse_args(argv)
    if a.cmd == "fit":
        fit(a.max_iter)
    else:
        res, hz = report()
        piv = res.pivot_table(index="fault_type", columns="score", values="auc", aggfunc="median")
        piv.loc["ALL (median over events)"] = res.groupby("score")["auc"].median()
        print("Median AUROC per event, fault window (detectable .. +6 h) vs pooled healthy minutes of the held-out pumps:")
        print(piv.round(3).to_string())
        print("\nHealthy minutes, held-out pumps: median |z| and share |z| > 3 (ideal for a standard normal: 0.67, 0.003)")
        print(hz.groupby(["set", hz["deviation"].str[:2]])[["median_abs", "frac_gt3"]].median().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
