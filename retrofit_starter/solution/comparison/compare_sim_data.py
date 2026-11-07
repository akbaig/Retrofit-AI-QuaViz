#!/usr/bin/env python3
"""Score frozen final model, fleet detector and baselines on unified sim_data.

Evaluates each component separately and as one pooled aggregate:

  fleets   sim_data/fleets/s0200..s0212
  stress   sim_data/stress
  astra    sim_data/astra/holdout
  all      concatenation of the three

  python compare_sim_data.py
  python compare_sim_data.py --sim-data ../../sim_data --out results/sim_data
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FLEET = HERE.parent / "fleet_detector"
CHALLENGE = HERE.parents[1]
sys.path.insert(0, str(FLEET))
sys.path.insert(0, str(CHALLENGE / "src"))

from quaviz_detector import data, harness, stage2  # noqa: E402
from quaviz_detector.methods import METHODS  # noqa: E402
from quaviz_detector.paths import OFFICIAL_DATA  # noqa: E402
from quaviz_detector.scorer import score_assets  # noqa: E402

COLS = ["composite_score", "event_f1", "event_recall", "alert_precision", "fault_type_recall",
        "class_recall", "timeliness", "false_alerts_per_asset_day", "events_scored"]
DEFAULT_SIM = CHALLENGE / "sim_data"
DEFAULT_OUT = HERE / "results" / "sim_data"
ASTRA_DIR = HERE.parent / "final_model"


def astra_dataset(path: Path) -> data.Dataset:
    ds = data.load(path, name="astra")
    ds.events = pd.read_csv(path / "events.csv")
    return ds


def infer_root(path: Path) -> Path:
    """Directory with assets.csv / features for final-model infer.py."""
    pub = path / "public"
    return pub if pub.is_dir() else path


def load_components(sim_data: Path) -> list[tuple[str, data.Dataset, Path]]:
    fleets_dir = sim_data / "fleets"
    fleet_paths = sorted(fleets_dir.glob("s0*"))
    if not fleet_paths:
        raise FileNotFoundError(f"No fleets under {fleets_dir}")
    fleet_sets = [data.load(p, name=p.name) for p in fleet_paths]
    fleets = data.concat(fleet_sets, name="fleets")
    stress = data.load(sim_data / "stress", name="stress")
    astra = astra_dataset(sim_data / "astra" / "holdout")
    return [
        ("fleets", fleets, fleets_dir),
        ("stress", stress, sim_data / "stress"),
        ("astra", astra, sim_data / "astra" / "holdout"),
    ]


def run_astra(data_dir: Path, out: Path) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    cmd = [sys.executable, "infer.py", "--model", "model.joblib", "--policy", "policy.json",
           "--data", str(data_dir), "--output", str(out)]
    env = os.environ.copy()
    env["QUAVIZ_CHALLENGE_DIR"] = str(CHALLENGE)
    subprocess.run(cmd, cwd=ASTRA_DIR, check=True, env=env)
    return pd.read_csv(out)


def predict_astra_component(label: str, root: Path, out_dir: Path) -> pd.DataFrame:
    if label == "fleets":
        parts = []
        for fleet in sorted(root.glob("s0*")):
            parts.append(run_astra(infer_root(fleet), out_dir / f"astra_pred_{fleet.name}.csv"))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return run_astra(infer_root(root), out_dir / f"astra_pred_{label}.csv")


def predict_method(method, name: str, ds: data.Dataset, ids: list[str]) -> pd.DataFrame:
    # New sim_data fleets/stress/astra are not in the stage-2 cache (old seeds 101-113).
    # QuaViz therefore builds stage-2 tables on the fly; baselines read features/alarms directly.
    if name == "quaviz":
        models: dict = {}
        tab = pd.concat([stage2.pump_table(ds, aid, models) for aid in ids], ignore_index=True)
        return method.predict_table(tab)
    return method.predict(ds, ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim-data", type=Path, default=DEFAULT_SIM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--methods", nargs="+", default=["astra-best", "quaviz", "b0", "b1", "b2", "b3"])
    ap.add_argument("--reuse", action="store_true", help="Reprint saved comparison.csv only.")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if a.reuse and (out / "comparison.csv").exists():
        print(pd.read_csv(out / "comparison.csv").round(3).to_string(index=False))
        return 0

    components = load_components(a.sim_data.resolve())
    all_ds = data.concat([ds for _, ds, _ in components], name="all")
    sets = [(label, ds, ds.ids()) for label, ds, _ in components] + [("all", all_ds, all_ds.ids())]

    official = data.load(OFFICIAL_DATA)
    train_ids = official.ids("train")

    preds: dict[tuple[str, str], pd.DataFrame] = {}

    # Final model: per-component, then concat for aggregate.
    if "astra-best" in a.methods:
        part_preds = {}
        for label, ds, root in components:
            print(f"astra-best infer on {label}…", flush=True)
            p = predict_astra_component(label, root, out)
            part_preds[label] = p
            preds[("astra-best", label)] = p
        preds[("astra-best", "all")] = pd.concat(part_preds.values(), ignore_index=True)

    # Fleet detector + baselines: fit once on official train.
    for name in a.methods:
        if name == "astra-best":
            continue
        print(f"fit {name} on official train…", flush=True)
        m = METHODS[name]()
        m.fit(official.subset(train_ids), train_ids)
        part_preds = {}
        for label, ds, _ in components:
            print(f"  predict {name} on {label} ({len(ds.ids())} assets)…", flush=True)
            p = predict_method(m, name, ds, ds.ids())
            part_preds[label] = p
            preds[(name, label)] = p
        preds[(name, "all")] = pd.concat(
            [p for p in part_preds.values() if len(p)], ignore_index=True
        ) if any(len(p) for p in part_preds.values()) else pd.DataFrame()

    rows, per_event = [], []
    for (method, label), p in preds.items():
        ds, ids = next((d, i) for l, d, i in sets if l == label)
        print(f"score {method} / {label}…", flush=True)
        r = score_assets(p, ds, ids, label)
        rows.append({"method": method, "set": label, **{c: r[c] for c in COLS}})
        per_event.append(pd.DataFrame(r["per_event"]).assign(method=method, set=label))
        safe = label.replace(" ", "_")
        p.to_csv(out / f"predictions_{method}_{safe}.csv", index=False)

    s = pd.DataFrame(rows)
    pe = pd.concat(per_event, ignore_index=True) if per_event else pd.DataFrame()
    s.to_csv(out / "comparison.csv", index=False)
    pe.to_csv(out / "per_event.csv", index=False)
    by_type = pe.groupby(["set", "fault_type", "method"], dropna=False).agg(
        events=("detected", "size"), detected=("detected", "sum"),
        type_correct=("type_correct", "sum")).reset_index() if len(pe) else pd.DataFrame()
    by_type.to_csv(out / "per_type.csv", index=False)

    parts = ["# Models on unified sim_data (official scorer)", ""]
    for label, g in s.groupby("set", sort=False):
        parts += [f"## {label}", "", harness._md(g.drop(columns="set").round(3)), ""]
    if len(by_type):
        parts += ["## Per fault type", "", harness._md(by_type), ""]
    (out / "comparison.md").write_text("\n".join(parts), encoding="utf-8")
    print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
