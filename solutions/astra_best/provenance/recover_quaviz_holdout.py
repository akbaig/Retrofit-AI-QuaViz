"""Finish reporting after the frozen evaluator's empty-event per-asset failure.

No fitting, model prediction, policy selection, model changes, or scorer changes.
Reuse the first saved frozen-model predictions and recompute only the three
predeclared deterministic B1 comparators. Aggregate scoring includes every asset.
"""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent / "quaviz-learned"
sys.path.insert(0, str(HERE))
from evaluate_frozen import verify_freeze, sha256, _write, METRICS
from paths import SOURCE

FREEZE = "a6edd3db339351f2d76a66f92921bd7f9604ae5d3b192b5f09e112e3ded2c75f"
manifest, _ = verify_freeze("final_ensemble", "runs/final_ensemble/policy_tuning", "FREEZE.json", FREEZE)
data = HERE / "synthetic_data/holdout"
dm = json.loads((data / "manifest.json").read_text())
assert dm["frozen_pipeline_manifest_sha256"] == FREEZE
assert sha256(data / "manifest.json") == "755927b41b7430f1985d83bb479c032c3f09cbca2a38511de3954eb575a1f925"
for name, digest in dm["artifacts"].items():
    assert sha256(data / name) == digest, name

import pandas as pd
from baseline_compare import evaluate_b1
sys.path.insert(0, str(SOURCE / "src"))
from retrofit_sim.scoring import score, load_predictions, bursts, _episodes

initial = HERE / "runs/final_ensemble/frozen_holdout"
out = initial / "completed_reports"
out.mkdir(exist_ok=False)
assets = pd.read_csv(data / "assets.csv")
events = pd.read_csv(data / "events.csv")
benign_path = data / "benign_windows.csv"
benign = pd.read_csv(benign_path) if benign_path.exists() else None
own_path = initial / "Quaviz_frozen_predictions.csv"
own = pd.read_csv(own_path)
saved = json.loads((initial / "Quaviz_frozen_report.json").read_text())
replayed = score(own, events, assets, benign)
assert replayed == saved, "Saved CSV does not reproduce the initial aggregate report"
configs = {
    "B1_published": {"k": 6.0, "persist_min": 15},
    "B1_development_selected": json.loads((HERE / manifest["b1_config_path"]).read_text()),
    "B1_validation_selected": {"k": 6.0, "persist_min": 5},
}
predictions = {"Quaviz_frozen": own}
reports = {"Quaviz_frozen": replayed}
for name, config in configs.items():
    predictions[name], reports[name] = evaluate_b1(data, config)

rows, by_asset, by_seed, per_event, per_type = [], [], [], [], []
for name, pred in predictions.items():
    pred.to_csv(out / f"{name}_predictions.csv", index=False)
    _write(out / f"{name}_report.json", reports[name])
    rows.append({"model": name, **{k: reports[name][k] for k in METRICS}})
    event_rows = pd.DataFrame(reports[name]["per_event"])
    per_event.extend({"model": name, **row} for row in reports[name]["per_event"])
    for kind, group in event_rows.groupby("fault_type"):
        per_type.append({"model": name, "fault_type": kind, "events": len(group),
                         "detected": int(group.detected.sum()), "correct_type": int(group.type_correct.sum())})
    for _, asset in assets.iterrows():
        scoped = assets.loc[assets.asset_id.eq(asset.asset_id)]
        base = {"model": name, "asset_id": asset.asset_id, "profile": asset.profile,
                "instrumentation_tier": asset.instrumentation_tier}
        if events.asset_id.eq(asset.asset_id).any():
            result = score(pred, events, scoped, benign)
            by_asset.append({**base, **{k: result[k] for k in METRICS}, "status": "official_score"})
        else:
            # The original scorer crashes on an empty event table with alerts.
            # Report these healthy assets explicitly without inventing events.
            selected = load_predictions(pred)
            selected = selected.loc[selected.asset_id.eq(asset.asset_id)]
            times = selected.alert_time_utc
            selected = selected.loc[times.ge(pd.Timestamp(asset.window_start_utc)) & times.lt(pd.Timestamp(asset.window_end_utc))]
            b = bursts(selected)
            by_asset.append({**base, "status": "eventless_asset_original_scorer_limitation",
                             "events_scored": 0, "alerts_scored": len(selected),
                             "false_alert_episodes": _episodes(b.alert_time_utc),
                             "excess_repeat_episodes": 0})
    for seed in dm["seeds"]:
        scoped = assets.loc[assets.asset_id.str.contains(f"-{seed}-", regex=False)]
        result = score(pred, events, scoped, benign)
        by_seed.append({"model": name, "seed": seed, **{k: result[k] for k in METRICS}})

for filename, content in (("comparison.csv", rows), ("per_asset.csv", by_asset),
                          ("per_seed.csv", by_seed), ("per_event.csv", per_event), ("per_type.csv", per_type)):
    pd.DataFrame(content).to_csv(out / filename, index=False)
verify_freeze("final_ensemble", "runs/final_ensemble/policy_tuning", "FREEZE.json", FREEZE)
for name, digest in dm["artifacts"].items():
    assert sha256(data / name) == digest, name
_write(out / "evaluation.json", {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "freeze_sha256": FREEZE, "holdout_manifest_sha256": sha256(data / "manifest.json"),
    "unchanged_model_sha256": sha256(HERE / manifest["model_path"]),
    "original_saved_prediction_sha256": sha256(own_path),
    "initial_aggregate_report_exactly_reproduced": True,
    "baseline_configs": configs, "reports": reports,
    "reporting_recovery": "The frozen evaluator completed aggregate scoring and input re-verification, then failed while producing a supplemental per-asset table for an eventless healthy asset. This report reuses its saved model predictions. Only the same three predeclared B1 configurations were recomputed. All aggregate and per-seed scores use the untouched official scorer and include all healthy assets. Per-asset official composites are unavailable for eventless assets; their false-alert counts remain reported. No model, features, policies, seed plans or original scorer changed.",
    "interpretation": "One frozen public-simulator holdout, not a real-plant or private-test evaluation.",
})
print(pd.DataFrame(rows)[["model", "composite_score", "event_recall", "fault_type_recall", "false_alert_episodes", "excess_repeat_episodes"]].to_string(index=False))
print(pd.DataFrame(by_seed)[["model", "seed", "composite_score"]].to_string(index=False))
