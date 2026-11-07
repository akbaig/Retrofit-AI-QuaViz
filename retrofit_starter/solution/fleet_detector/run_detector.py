#!/usr/bin/env python3
"""Run the fleet detector and the reference baselines through the same harness and official scorer.

  python run_detector.py
  python run_detector.py --methods quaviz --no-lopo --no-submission
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

import pandas as pd  # noqa: E402

from quaviz_detector import data, harness  # noqa: E402
from quaviz_detector.methods import QuaViz  # noqa: E402
from quaviz_detector.paths import RESULTS, SIM_FLEETS  # noqa: E402
from quaviz_detector.stage1 import HOLDOUT_SEEDS  # noqa: E402

COLS = ["composite_score", "event_f1", "fault_type_recall", "class_recall", "timeliness", "false_alerts_per_asset_day",
        "events_scored"]


def comparison(methods) -> str:
    rows = []
    for m in methods:
        p = RESULTS / m / "summary.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    s = pd.concat(rows)
    s["split"] = s["split"].str.replace("sim:", "held-out sim ")
    parts = ["# Comparison on the official scorer", ""]
    for split, g in s.groupby("split", sort=False):
        parts += [f"## {split}", "", harness._md(g[["method"] + COLS].round(3)), ""]
    return "\n".join(parts)


def submission(config: str = "quaviz") -> pd.DataFrame:
    official = data.load()
    m = QuaViz(config).fit(official, official.ids("train") + official.ids("validation"))
    p = m.predict(official, official.ids("test"))
    out = RESULTS / config
    out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "submission_test.csv", index=False)
    print(f"submission: {len(p)} alerts on {p['asset_id'].nunique()} test pumps -> {out / 'submission_test.csv'}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", nargs="*", default=["quaviz", "b0", "b1", "b2", "b3"])
    ap.add_argument("--no-lopo", action="store_true")
    ap.add_argument("--no-submission", action="store_true")
    a = ap.parse_args(argv)
    sims = [SIM_FLEETS / f"s{s:04d}" for s in HOLDOUT_SEEDS]
    for m in a.methods:
        res = harness.run(m, lopo=not a.no_lopo, eval_sim=sims, out=RESULTS / m)
        print(res["summary"][["method", "split"] + COLS].round(3).to_string(index=False), flush=True)
    every = [m for m in ["quaviz", "quaviz_nodev", "quaviz_simonly", "quaviz_officialonly", "b0", "b1", "b2", "b3"]
             if (RESULTS / m / "summary.csv").exists()]
    (RESULTS / "comparison.md").write_text(comparison(every), encoding="utf-8")
    if not a.no_submission and "quaviz" in a.methods:
        submission()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
