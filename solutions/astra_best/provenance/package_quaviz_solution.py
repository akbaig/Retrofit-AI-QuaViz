"""Assemble a portable delivery; never modify the frozen experiment."""
from pathlib import Path
import hashlib
import json
import shutil

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "quaviz-learned"
FINAL = SOURCE / "runs/final_ensemble"
DEST = BASE / "QuaViz_Learned_Solution"
DEST.mkdir(exist_ok=False)

def copy(source, destination):
    target = DEST / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

for path in SOURCE.glob("*.py"):
    copy(path, path.name)
for name in ("README.txt", "REPRODUCE.txt", "RESULTS.txt", "requirements.txt"):
    copy(SOURCE / name, name)
copy(FINAL / "model.joblib", "model.joblib")
copy(FINAL / "policy_tuning/policy.json", "policy.json")
for path in (SOURCE / "tests").glob("*.py"):
    copy(path, Path("tests") / path.name)
for name in ("original_validation_predictions.csv", "original_validation_report.json",
             "development_predictions.csv", "development_report.json", "test_predictions.csv"):
    copy(FINAL / name, Path("results") / name)
shutil.copytree(FINAL / "policy_tuning", DEST / "results/development_search")
shutil.copytree(SOURCE / "baseline_results", DEST / "results/baseline_development")
shutil.copytree(FINAL / "frozen_holdout/completed_reports", DEST / "results/holdout")

for path in SOURCE.glob("*_notes.txt"):
    copy(path, Path("provenance/audits") / path.name)
for name in ("FREEZE.json", "FREEZE.sha256", "DESIGN.md", "EXPERIMENT_LOG.md", "ensemble_prediction_audit.json"):
    copy(SOURCE / name, Path("provenance") / name)
for name in ("composition.json", "selection_provenance.json", "verification.json"):
    copy(FINAL / name, Path("provenance/final_model") / name)
for path in (FINAL / "frozen_holdout").glob("*.*"):
    copy(path, Path("provenance/initial_holdout_output") / path.name)
copy(BASE / "recover_quaviz_holdout.py", "provenance/recover_quaviz_holdout.py")
copy(Path(__file__), "provenance/package_quaviz_solution.py")

for path in (SOURCE / "synthetic_data").rglob("*.json"):
    copy(path, Path("provenance") / path.relative_to(SOURCE))
for path in (SOURCE / "synthetic_data").glob("*/*.csv"):
    # Small labels/specifications/context permit score and scope auditing.
    # No raw telemetry, minute feature tables or waveform arrays are copied.
    copy(path, Path("provenance") / path.relative_to(SOURCE))
for run in (SOURCE / "runs").iterdir():
    if not run.is_dir() or run.name == "final_ensemble":
        continue
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".txt"):
            copy(path, Path("provenance/experiments") / path.relative_to(SOURCE / "runs"))

old = json.loads((BASE / "quaviz-causal-benchmark/results/validation_report.json").read_text())
(DEST / "results/published_B3_validation_report.json").write_text(
    json.dumps(old["published_B3_IsolationForest"], indent=2) + "\n")
(DEST / "provenance/initial_holdout_failure.txt").write_text('''Original frozen evaluation command:
python evaluate_frozen.py --run final_ensemble --policy-dir runs/final_ensemble/policy_tuning --freeze-manifest FREEZE.json --freeze-sha256 a6edd3db339351f2d76a66f92921bd7f9604ae5d3b192b5f09e112e3ded2c75f

Exit 1, after aggregate scoring and successful input/hash re-verification.
Traceback excerpt:
  evaluate_frozen.py:237 evaluate(...)
  evaluate_frozen.py:207 r = score(pred, events, scoped, benign)
  retrofit_sim/scoring.py:147 cand = e_a[(e_a["w_start"] <= t) & (t <= e_a["w_end"])]
  pandas/core/ops/array_ops.py:344 res_values = op(lvalues, rvalues)
TypeError: '<=' not supported between instances of 'numpy.ndarray' and 'Timestamp'

The supplemental per-asset call has zero event rows on a healthy asset; the
original scorer creates float-typed empty event-window columns and fails when
comparing them with an alert timestamp. The initial aggregate report and
predictions are preserved here. The separate reporting continuation reuses
those predictions, exactly reproduces their aggregate score, evaluates the
same three fixed baselines, and preserves all healthy assets in aggregate
scores. No original scoring code or frozen model/policy was modified.
''')
(DEST / "provenance/final_test_evidence.json").write_text(json.dumps({
    "original_workspace_full_suite": {"passed": 40, "warnings": 11, "seconds": 6.24,
        "reviewer": "Independent learned_eval_review agent, 2026-11-07"},
    "coverage": ["observation availability and missingness", "raw and fitted-model prefix invariance",
                 "future-mutation isolation", "sensor routing", "alert persistence",
                 "training reproduction CLI", "existing artifact preservation"]
}, indent=2) + "\n")
print(f"Staged {sum(p.is_file() for p in DEST.rglob('*'))} files at {DEST}")
print("ZIP and portable manifest follow the extracted-layout inference smoke test.")
