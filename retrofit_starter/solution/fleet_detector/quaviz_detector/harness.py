"""Evaluation harness on the official scorer.

Protocols
  official  fit on the official train pumps (+ optional simulated fleets), score train and validation separately,
            exactly as the published baselines were evaluated.
  lopo      leave-one-pump-out over the 9 labeled official pumps; predictions are pooled and scored together.
  sim       fit as in ``official``, score on simulated fleets (generalization to unseen configurations).

    python -m quaviz_detector.harness --method b1 --lopo --eval-sim build/sim/s0101 --out results/b1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import data
from .methods import METHODS
from .paths import RESULTS
from .scorer import score_assets

METRICS = ["composite_score", "event_f1", "event_recall", "alert_precision", "false_alerts_per_asset_day", "timeliness",
           "fault_type_recall", "class_recall"]
COUNTS = ["events_scored", "true_positive_events", "false_alert_episodes", "excess_repeat_episodes", "alert_bursts"]


def _fit_set(official, ids, sims):
    parts = [official.subset(ids)] + sims
    ds = data.concat(parts, name="fit") if sims else parts[0]
    return ds, list(ids) + [a for s in sims for a in s.ids()]


def _record(rows, per_event, label, r, ds, method):
    rows.append({"method": method.name, "split": label, **{m: r[m] for m in METRICS}, **{c: r[c] for c in COUNTS}})
    pe = pd.DataFrame(r["per_event"])
    if len(pe):
        meta = ds.assets[["asset_id", "profile", "control", "instrumentation_tier"]]
        per_event.append(pe.merge(meta, on="asset_id", how="left").assign(split=label))


def run(method_name: str, lopo: bool = False, train_sim=(), eval_sim=(), out=None, write_test: bool = True) -> dict:
    t0 = time.time()
    make = METHODS[method_name]
    official = data.load()
    sims_train = [data.load(p) for p in train_sim]
    sims_eval = [data.load(p) for p in eval_sim]
    train_ids, val_ids, test_ids = official.ids("train"), official.ids("validation"), official.ids("test")
    rows, per_event, preds = [], [], {}

    method = make()
    fit_ds, fit_ids = _fit_set(official, train_ids, sims_train)
    method.fit(fit_ds, fit_ids)
    params = {"official": method.describe()}
    for split, ids in (("train", train_ids), ("validation", val_ids)):
        p = method.predict(official, ids)
        preds[split] = p
        _record(rows, per_event, split, score_assets(p, official, ids, split), official, method)
    if write_test:
        preds["test"] = method.predict(official, test_ids)
    for s in sims_eval:
        p = method.predict(s, s.ids())
        _record(rows, per_event, f"sim:{s.name}", score_assets(p, s, s.ids(), f"sim:{s.name}"), s, method)

    if lopo:
        labeled = train_ids + val_ids
        fold_preds, params["lopo"] = [], {}
        for aid in labeled:
            others = [a for a in labeled if a != aid]
            m = make()
            fds, fids = _fit_set(official, others, sims_train)
            m.fit(fds, fids)
            params["lopo"][aid] = m.describe()
            fold_preds.append(m.predict(official, [aid]))
        p = pd.concat(fold_preds, ignore_index=True)
        preds["lopo"] = p
        _record(rows, per_event, "lopo", score_assets(p, official, labeled, "lopo"), official, method)

    summary = pd.DataFrame(rows)
    pe = pd.concat(per_event, ignore_index=True) if per_event else pd.DataFrame()
    result = {"method": method_name, "params": params, "summary": summary, "per_event": pe, "predictions": preds,
              "train_sim": [str(p) for p in train_sim], "eval_sim": [str(p) for p in eval_sim],
              "seconds": round(time.time() - t0, 1)}
    if out:
        write(result, Path(out))
    return result


def breakdown(pe: pd.DataFrame, by) -> pd.DataFrame:
    g = pe.groupby(["split"] + by)
    return pd.DataFrame({"events": g.size(), "detected": g["detected"].sum(), "type_correct": g["type_correct"].sum(),
                         "class_correct": g["class_correct"].sum()}).reset_index()


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write(result: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    s, pe = result["summary"], result["per_event"]
    s.to_csv(out / "summary.csv", index=False)
    if len(pe):
        pe.to_csv(out / "per_event.csv", index=False)
    for split, p in result["predictions"].items():
        p.to_csv(out / f"predictions_{split.replace(':', '_')}.csv", index=False)
    parts = [f"# {result['method']}", "", f"Runtime {result['seconds']} s. Parameters: `{json.dumps(result['params'])}`", "",
             "## Scores (official scorer)", "", _md(s.round(3)), ""]
    if len(pe):
        parts += ["## By control type", "", _md(breakdown(pe, ["control"])), "",
                  "## By instrumentation tier", "", _md(breakdown(pe, ["instrumentation_tier"])), "",
                  "## By true fault type", "", _md(breakdown(pe, ["fault_type"])), ""]
    (out / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({k: result[k] for k in ("method", "params", "train_sim", "eval_sim", "seconds")},
                                             indent=2, default=str), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m quaviz_detector.harness", description=__doc__.split("\n\n")[0])
    p.add_argument("--method", default="b1", choices=sorted(METHODS))
    p.add_argument("--lopo", action="store_true", help="Also run leave-one-pump-out over the 9 labeled pumps.")
    p.add_argument("--train-sim", nargs="*", default=[], help="Simulated fleet folders added to every fit set.")
    p.add_argument("--eval-sim", nargs="*", default=[], help="Simulated fleet folders to score on.")
    p.add_argument("--out", default=None, help=f"Output folder (default {RESULTS}/<method>).")
    a = p.parse_args(argv)
    res = run(a.method, a.lopo, a.train_sim, a.eval_sim, a.out or RESULTS / a.method)
    print(res["summary"][["method", "split"] + METRICS + ["events_scored"]].round(3).to_string(index=False))
    print(f"params: {json.dumps(res['params'], default=str)}")
    print(f"runtime {res['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
