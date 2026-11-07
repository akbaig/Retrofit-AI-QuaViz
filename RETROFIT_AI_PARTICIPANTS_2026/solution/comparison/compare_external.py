#!/usr/bin/env python3
"""Score the fleet detector, the reference baselines and the frozen final model on the same data.

Sets:
  validation      PUMP-08, PUMP-09
  our holdout     simulated fleets s0112, s0113 (with snapshots for the final model)
  astra holdout   final model's frozen holdout (regenerated from its seeds)

  python compare_external.py \\
      --astra-dir ../final_model \\
      --our-holdout-snap ../../../build/sim_snap \\
      --astra-holdout ../../../build/astra_synth/holdout
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLEET = HERE.parent / "fleet_detector"
CHALLENGE = HERE.parents[1]
sys.path.insert(0, str(FLEET))
sys.path.insert(0, str(CHALLENGE / "src"))

import pandas as pd  # noqa: E402

from quaviz_detector import data, harness, stage2  # noqa: E402
from quaviz_detector.methods import METHODS  # noqa: E402
from quaviz_detector.paths import OFFICIAL_DATA, SIM_FLEETS  # noqa: E402
from quaviz_detector.scorer import score_assets  # noqa: E402
from quaviz_detector.stage1 import HOLDOUT_SEEDS  # noqa: E402

COLS = ["composite_score", "event_f1", "event_recall", "alert_precision", "fault_type_recall", "class_recall",
        "timeliness", "false_alerts_per_asset_day", "events_scored"]
OUT = HERE / "results"


def astra_dataset(path: Path) -> data.Dataset:
    ds = data.load(path, name="astra_holdout")
    ds.events = pd.read_csv(path / "events.csv")
    return ds


def run_astra(astra_dir: Path, data_dir: Path, out: Path, split=None) -> pd.DataFrame:
    if not out.exists():
        cmd = [sys.executable, "infer.py", "--model", "model.joblib", "--policy", "policy.json",
               "--data", str(data_dir), "--output", str(out)] + (["--split", split] if split else [])
        env = os.environ.copy()
        env["QUAVIZ_CHALLENGE_DIR"] = str(CHALLENGE)
        subprocess.run(cmd, cwd=astra_dir, check=True, env=env)
    return pd.read_csv(out)


def check_same(a: data.Dataset, b: data.Dataset) -> None:
    key = lambda e: e.sort_values("event_id").reset_index(drop=True)
    if not (a.features.equals(b.features) and a.context.equals(b.context) and key(a.events).equals(key(b.events))):
        raise ValueError(f"{b.name}: regenerated fleet differs from the one the fleet detector was evaluated on")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--astra-dir", type=Path, default=HERE.parent / "final_model")
    ap.add_argument("--our-holdout-snap", type=Path, default=CHALLENGE.parent / "build" / "sim_snap")
    ap.add_argument("--astra-holdout", type=Path, default=CHALLENGE.parent / "build" / "astra_synth" / "holdout")
    ap.add_argument("--reuse", action="store_true", help="Only rebuild the comparison table from saved predictions.")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.reuse and (OUT / "comparison.csv").exists():
        print(pd.read_csv(OUT / "comparison.csv").round(3).to_string(index=False))
        return 0

    official = data.load()
    train_ids, val_ids = official.ids("train"), official.ids("validation")
    ours = [data.load(SIM_FLEETS / f"s{s:04d}") for s in HOLDOUT_SEEDS]
    for d in ours:
        check_same(d, data.load(a.our_holdout_snap / d.name))
    ah = astra_dataset(a.astra_holdout.resolve())
    sets = [("validation", official, val_ids)] + [(f"our holdout {d.name}", d, d.ids()) for d in ours] + \
           [("astra holdout", ah, ah.ids())]

    preds = {}
    for name in ["quaviz", "b0", "b1", "b2", "b3"]:
        m = METHODS[name]()
        m.fit(official.subset(train_ids), train_ids)
        for label, ds, ids in sets:
            if label == "astra holdout" and name == "quaviz":
                models: dict = {}
                p = m.predict_table(pd.concat([stage2.pump_table(ds, aid, models) for aid in ids]))
            else:
                p = m.predict(ds, ids)
            preds[(name, label)] = p
    ad = a.astra_dir.resolve()
    preds[("astra-best", "validation")] = run_astra(ad, OFFICIAL_DATA, OUT / "astra_validation.csv", "validation")
    for d in ours:
        preds[("astra-best", f"our holdout {d.name}")] = run_astra(
            ad, (a.our_holdout_snap / d.name / "public").resolve(), OUT / f"astra_{d.name}.csv")
    preds[("astra-best", "astra holdout")] = run_astra(ad, a.astra_holdout.resolve(), OUT / "astra_astra_holdout.csv")

    rows, per_event = [], []
    for (method, label), p in preds.items():
        ds, ids = next((d, i) for l, d, i in sets if l == label)
        r = score_assets(p, ds, ids, label)
        rows.append({"method": method, "set": label, **{c: r[c] for c in COLS}})
        per_event.append(pd.DataFrame(r["per_event"]).assign(method=method, set=label))
        p.to_csv(OUT / f"predictions_{method}_{label.replace(' ', '_')}.csv", index=False)
    s = pd.DataFrame(rows)
    pe = pd.concat(per_event, ignore_index=True)
    s.to_csv(OUT / "comparison.csv", index=False)
    pe.to_csv(OUT / "per_event.csv", index=False)
    by_type = pe.groupby(["set", "fault_type", "method"]).agg(
        events=("detected", "size"), detected=("detected", "sum"), type_correct=("type_correct", "sum")).reset_index()
    parts = ["# Fleet detector, baselines and final model on the same data (official scorer)", ""]
    for label, g in s.groupby("set", sort=False):
        parts += [f"## {label}", "", harness._md(g.drop(columns="set").round(3)), ""]
    parts += ["## Per fault type", "", harness._md(by_type), ""]
    (OUT / "comparison.md").write_text("\n".join(parts), encoding="utf-8")
    print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
