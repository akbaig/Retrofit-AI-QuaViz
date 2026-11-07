"""Bounded development-only threshold tuning for an existing probability run.

This uses the production alerts.make_alerts and the unchanged official scorer.
It first explores global threshold/persistence, then makes a fixed-order, bounded
coordinate search over type thresholds, global persistence and recovery. Both development
sets receive equal weight. The selected development score is not an estimate of
unseen performance; only a subsequently frozen independent holdout can do that.

Example:
  python tune_policy.py --name hist_01
  python tune_policy.py --name routed_01 --input calibrated_probabilities.parquet

--input is a filename suffix: '<dataset>_<suffix>'. Only original_validation and
development are read, with no holdout option. Existing policy.json is untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from alerts import AlertPolicy, make_alerts
from learning import FAULTS
from run import HERE, SOURCE, prepared
from retrofit_sim.scoring import score


DATASETS = ("original_validation", "development")
THRESHOLDS = (.005, .01, .02, .04, .08, .15, .25, .4, .6, .8, .95)
PERSISTENCES = (3, 5, 10, 15)
RECOVERIES = (60, 240, 720)
METRICS = ("composite_score", "event_f1", "event_recall", "alert_precision", "timeliness",
           "fault_type_recall", "class_recall", "events_scored", "true_positive_events",
           "false_alert_episodes", "excess_repeat_episodes", "alerts_scored")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_clean_json(value), indent=2, allow_nan=False) + "\n")


def _policy_key(policy):
    return (tuple(float(policy.thresholds.get(f, policy.threshold)) for f in FAULTS),
            policy.persistence, policy.recovery, policy.reminder)


def tune(name, input_suffix="probabilities.parquet", output=None, max_passes=2):
    if not 0 <= max_passes <= 3:
        raise ValueError("Use at most three passes to keep the search bounded")
    if Path(input_suffix).name != input_suffix:
        raise ValueError("Input suffix must be a filename, not a directory")
    run_dir = HERE / "runs" / name
    output = Path(output) if output else run_dir / "policy_tuning"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Retain earlier trials: select an empty output directory")
    probability_paths = {d: run_dir / f"{d}_{input_suffix.lstrip('_')}" for d in DATASETS}
    datasets = {d: prepared(d) for d in DATASETS}
    probabilities = {d: pd.read_parquet(probability_paths[d]) for d in DATASETS}
    for dataset in DATASETS:
        meta, _, assets, _, _ = datasets[dataset]
        p = probabilities[dataset]
        if len(meta) != len(p) or set(FAULTS) - set(p):
            raise ValueError(f"Probability rows/columns do not align for {dataset}")
        if not np.isfinite(p[FAULTS].to_numpy()).all() or ((p[FAULTS] < 0) | (p[FAULTS] > 1)).any().any():
            raise ValueError(f"Expected finite [0,1] scores for {dataset}")
        if dataset == "original_validation":
            assert set(assets.asset_id) == {"PUMP-08", "PUMP-09"}
        else:
            assert set(assets["split"]) == {"development"}
    source_paths = [HERE / p for p in ("alerts.py", "learning.py", "features.py", "physics_features.py", "run.py", "tune_policy.py")]
    source_paths.append(SOURCE / "src/retrofit_sim/scoring.py")
    provenance = {p.name: _sha(p) for p in source_paths}
    _write(output / "search_plan.json", {
        "run": name, "development_datasets": list(DATASETS), "objective": "Equal mean official composite score",
        "threshold_grid": list(THRESHOLDS), "global_persistence_grid": list(PERSISTENCES),
        "coordinate_order": FAULTS + ["global_persistence", "recovery"], "max_coordinate_passes": max_passes,
        "initial_recovery": 60, "recovery_grid": list(RECOVERIES), "reminder": 1440,
        "initial_grid_policies": len(THRESHOLDS)*len(PERSISTENCES),
        "maximum_proposed_policies": len(THRESHOLDS)*len(PERSISTENCES)+max_passes*(len(FAULTS)*len(THRESHOLDS)+len(PERSISTENCES)+len(RECOVERIES)),
        "source_sha256": provenance, "probability_sha256": {d: _sha(p) for d, p in probability_paths.items()},
        "no_holdout_access": True,
        "interpretation": "Adaptive fitting to both declared development sets. This score is not independent generalization evidence.",
    })
    started = time.perf_counter()
    evaluated, flat_rows, cache, trace = [], [], {}, []
    best_seen = -np.inf

    def evaluate(policy, phase):
        nonlocal best_seen
        key = _policy_key(policy)
        if key in cache:
            result = cache[key]
            trace.append({"phase": phase, "trial_id": result["trial_id"], "cache_hit": True})
            return result
        trial_id = len(evaluated)
        reports = {}
        for dataset in DATASETS:
            meta, _, assets, events, benign = datasets[dataset]
            pred = make_alerts(meta, probabilities[dataset], policy)
            reports[dataset] = score(pred, events, assets, benign)
        objective = float(np.mean([r["composite_score"] for r in reports.values()]))
        result = {"trial_id": trial_id, "phase": phase, "policy": asdict(policy),
                  "selection_mean": objective,
                  "total_alerts": int(sum(r["alerts_scored"] for r in reports.values())),
                  "scores": {d: {k: r[k] for k in METRICS} for d, r in reports.items()},
                  "reports": reports, "elapsed_seconds": time.perf_counter()-started}
        cache[key] = result
        evaluated.append(result)
        trace.append({"phase": phase, "trial_id": trial_id, "cache_hit": False})
        for dataset, report in reports.items():
            flat_rows.append({"trial_id": trial_id, "phase": phase, "dataset": dataset,
                              "selection_mean": objective, "threshold": policy.threshold,
                              "persistence": policy.persistence, "recovery": policy.recovery,
                              "reminder": policy.reminder, "thresholds_json": json.dumps(policy.thresholds, sort_keys=True),
                              **{k: report[k] for k in METRICS}})
        with (output / "trials.jsonl").open("a") as handle:
            handle.write(json.dumps(_clean_json(result), allow_nan=False) + "\n")
        if objective > best_seen+1e-9:
            best_seen = objective
            print("trial", trial_id, phase, "mean", round(objective, 4),
                  {d: round(r["composite_score"], 4) for d, r in reports.items()}, flush=True)
        elif trial_id % 25 == 0:
            print("trial", trial_id, phase, "best_mean", round(best_seen, 4), flush=True)
        return result

    globals_ = [evaluate(AlertPolicy(threshold=t, persistence=p, recovery=60, reminder=1440), "global_grid")
                for t, p in itertools.product(THRESHOLDS, PERSISTENCES)]
    # Tie-breaking cannot trade away equal-mean score; it favors fewer alerts,
    # then higher thresholds and longer persistence within the global grid.
    current = max(globals_, key=lambda r: (r["selection_mean"], -r["total_alerts"],
                                           r["policy"]["threshold"], r["policy"]["persistence"]))
    global_best = current
    coordinate_steps = []
    for pass_number in range(1, max_passes+1):
        improved = False
        for fault in FAULTS:
            previous = current
            base = AlertPolicy(**current["policy"])
            alternatives = [current]
            for threshold in THRESHOLDS:
                thresholds = dict(base.thresholds)
                if threshold == base.threshold:
                    thresholds.pop(fault, None)
                else:
                    thresholds[fault] = threshold
                candidate = AlertPolicy(base.threshold, base.persistence, base.recovery, 1440, thresholds)
                alternatives.append(evaluate(candidate, f"pass{pass_number}:{fault}"))
            chosen = max(alternatives, key=lambda r: r["selection_mean"])
            # Keep the present setting on ties, avoiding gratuitous complexity.
            if chosen["selection_mean"] > current["selection_mean"]+1e-9:
                current = chosen
                improved = True
            coordinate_steps.append({"pass": pass_number, "coordinate": fault,
                "from_trial": previous["trial_id"], "to_trial": current["trial_id"],
                "accepted": current is not previous})
        previous = current
        base = AlertPolicy(**current["policy"])
        alternatives = [current] + [evaluate(AlertPolicy(base.threshold, p, base.recovery, 1440, dict(base.thresholds)),
                                            f"pass{pass_number}:global_persistence") for p in PERSISTENCES]
        chosen = max(alternatives, key=lambda r: r["selection_mean"])
        if chosen["selection_mean"] > current["selection_mean"]+1e-9:
            current = chosen
            improved = True
        coordinate_steps.append({"pass": pass_number, "coordinate": "global_persistence",
            "from_trial": previous["trial_id"], "to_trial": current["trial_id"], "accepted": current is not previous})
        previous = current
        base = AlertPolicy(**current["policy"])
        alternatives = [current] + [evaluate(AlertPolicy(base.threshold, base.persistence, recovery, 1440,
                                                         dict(base.thresholds)), f"pass{pass_number}:recovery")
                                     for recovery in RECOVERIES]
        chosen = max(alternatives, key=lambda r: r["selection_mean"])
        if chosen["selection_mean"] > current["selection_mean"]+1e-9:
            current = chosen
            improved = True
        coordinate_steps.append({"pass": pass_number, "coordinate": "recovery",
            "from_trial": previous["trial_id"], "to_trial": current["trial_id"], "accepted": current is not previous})
        if not improved:
            break
    policy = AlertPolicy(**current["policy"])
    selected_reports = {}
    for dataset in DATASETS:
        meta, _, assets, events, benign = datasets[dataset]
        pred = make_alerts(meta, probabilities[dataset], policy)
        report = score(pred, events, assets, benign)
        if report != current["reports"][dataset]:
            raise AssertionError("Selected policy did not reproduce its logged score")
        selected_reports[dataset] = report
        pred.to_csv(output / f"{dataset}_alerts.csv", index=False)
    pd.DataFrame(flat_rows).to_csv(output / "trials.csv", index=False)
    _write(output / "search_trace.json", trace)
    _write(output / "coordinate_steps.json", coordinate_steps)
    _write(output / "policy.json", asdict(policy))
    selection = {"run": name, "policy": asdict(policy), "selected_trial": current["trial_id"],
                 "selection_mean": current["selection_mean"], "reports": selected_reports,
                 "best_global_trial": global_best["trial_id"], "best_global_mean": global_best["selection_mean"],
                 "unique_evaluations": len(evaluated), "proposal_count": len(trace),
                 "elapsed_seconds": time.perf_counter()-started, "development_datasets": list(DATASETS),
                 "holdout_evaluated": False, "source_sha256": provenance,
                 "interpretation": "Selected on familiar validation and synthetic development; do not claim this adaptive score as unseen generalization."}
    _write(output / "selection.json", selection)
    print("Selected", asdict(policy), "mean", round(current["selection_mean"], 4),
          "trials", len(evaluated), "seconds", round(selection["elapsed_seconds"], 1), flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Existing run under experiments/quaviz-learned/runs")
    parser.add_argument("--input", "--input-suffix", dest="input_suffix", default="probabilities.parquet")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-passes", type=int, default=2)
    args = parser.parse_args()
    tune(args.name, args.input_suffix, args.output, args.max_passes)
