"""Give unchanged published B1 the same allowed development sets as Quaviz.

Selection uses the equal mean of public (familiar) validation and freshly
generated synthetic development. No published test asset is loaded. This file
does not create or inspect synthetic holdout data during selection.

evaluate_b1(data_dir, config) returns (predictions, official_score_report).
For an original data directory, config['split'] defaults to 'validation' and
only train/validation are allowed. For a generated data directory containing
events.csv, all explicitly supplied synthetic assets are evaluated.

The published first-24-hour normalization is preserved exactly, including its
early-window lookahead. This comparator is not represented as a causal model.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


from paths import HERE, SOURCE
sys.path.insert(0, str(SOURCE / "src"))
from retrofit_sim import baselines as B  # noqa: E402
from retrofit_sim.scoring import score  # noqa: E402

PUBLISHED_CONFIG = {"k": 6.0, "persist_min": 15}
K_GRID = (3., 4., 6., 8., 10.)
PERSIST_GRID = (5, 10, 15, 30)
FORBIDDEN_ASSETS = {"PUMP-10", "PUMP-11", "PUMP-12"}
METRICS = ("composite_score", "event_f1", "event_recall", "alert_precision",
           "fault_type_recall", "class_recall", "timeliness", "false_alerts_per_asset_day",
           "events_scored", "true_positive_events", "false_alert_episodes", "excess_repeat_episodes",
           "alerts_scored", "alert_bursts", "median_lead_time_h")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n")


def _parameters(config):
    k = float(config.get("k", PUBLISHED_CONFIG["k"]))
    persistence = config.get("persist_min", config.get("persist", PUBLISHED_CONFIG["persist_min"]))
    persist_min = int(persistence)
    if not np.isfinite(k) or k <= 0 or persist_min <= 0 or persist_min != float(persistence):
        raise ValueError("B1 k must be positive/finite and persistence a positive whole number of minutes")
    return k, persist_min


def _load(data_dir, config):
    data_dir = Path(data_dir).resolve()
    assets = pd.read_csv(data_dir / "assets.csv")
    synthetic = (data_dir / "events.csv").exists()
    if synthetic:
        if assets.asset_id.isin(FORBIDDEN_ASSETS).any() or assets["split"].eq("test").any():
            raise ValueError("Published test assets are outside this experiment")
        event_path = data_dir / "events.csv"
        benign_path = data_dir / "benign_windows.csv"
    else:
        split = config.get("split", "validation")
        if split not in ("train", "validation"):
            raise ValueError("Only published train or familiar validation may be evaluated")
        assets = assets.loc[assets["split"].eq(split)].copy()
        permitted = {f"PUMP-{i:02d}" for i in (range(1, 8) if split == "train" else (8, 9))}
        if not set(assets.asset_id).issubset(permitted):
            raise ValueError("Unexpected published split assets")
        event_path = data_dir / f"events_{split}.csv"
        benign_path = data_dir / "benign_windows_train_validation.csv"
    if assets.empty or assets.asset_id.duplicated().any():
        raise ValueError("Expected a non-empty set of unique asset identifiers")
    ids = assets.asset_id.tolist()
    # Parquet row filtering happens before any published test telemetry is read.
    features_path = data_dir / "features_1min.parquet"
    features = pd.read_parquet(features_path, filters=[("asset_id", "in", ids)])
    if set(features.asset_id) - set(ids):
        raise AssertionError("Feature scope escaped requested assets")
    events = pd.read_csv(event_path)
    events = events.loc[events.asset_id.isin(ids)].copy()
    benign = pd.read_csv(benign_path) if benign_path.exists() else None
    if benign is not None:
        benign = benign.loc[benign.asset_id.isin(ids)].copy()
    z = B.zscores(B.regime_frame(features, assets))
    input_paths = [data_dir / "assets.csv", features_path, event_path]
    if benign_path.exists():
        input_paths.append(benign_path)
    provenance = {"data_dir": str(data_dir), "assets": ids,
                  "input_sha256": {p.name: _sha(p) for p in input_paths}}
    return z, assets, events, benign, provenance


def _evaluate_prepared(prepared, config):
    z, assets, events, benign, _ = prepared
    k, persist_min = _parameters(config)
    predictions = B.b1_threshold(z, k=k, persist=persist_min)
    report = score(predictions, events, assets, benign)
    return predictions, report


def evaluate_b1(data_dir, config):
    """Return unchanged B1 predictions and official score for one scoped dataset.

    Accepts {'k': float, 'persist_min': int}; optional 'split' selects train or
    validation for the original dataset only. The caller controls when a frozen
    synthetic holdout is supplied. No state or files are mutated by this call.
    """
    return _evaluate_prepared(_load(data_dir, config), config)


def select_b1(original_data=SOURCE / "data/full", development=HERE / "synthetic_data/development",
              output=HERE / "baseline_results"):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise FileExistsError("Retain previous runs: choose a new output directory")
    datasets = {
        "original_validation": _load(original_data, {"split": "validation"}),
        "development": _load(development, {}),
    }
    # This assertion protects the selection entrypoint from accidentally being
    # pointed at a generated holdout; evaluate_b1 is the later evaluation API.
    if not set(datasets["development"][1]["split"]) <= {"development"}:
        raise ValueError("Selection requires synthetic development, never holdout")
    trials, all_reports, predictions_by_policy = [], {}, {}
    rankings = []
    for ordinal, (k, persistence) in enumerate(itertools.product(K_GRID, PERSIST_GRID)):
        config = {"k": k, "persist_min": persistence}
        policy_id = f"k{k:g}_p{persistence}"
        group_reports, group_predictions = {}, {}
        for name, prepared in datasets.items():
            predictions, report = _evaluate_prepared(prepared, config)
            group_reports[name], group_predictions[name] = report, predictions
            trials.append({"policy_id": policy_id, "dataset": name, **config,
                           **{key: report[key] for key in METRICS}})
        mean = float(np.mean([r["composite_score"] for r in group_reports.values()]))
        rankings.append({"policy_id": policy_id, "selection_mean": mean, "grid_ordinal": ordinal, **config})
        all_reports[policy_id], predictions_by_policy[policy_id] = group_reports, group_predictions
        print(policy_id, "equal_mean", round(mean, 4),
              {name: round(r["composite_score"], 4) for name, r in group_reports.items()}, flush=True)
    # Exact ties retain the first predeclared grid position.
    ranking = pd.DataFrame(rankings).sort_values(["selection_mean", "grid_ordinal"], ascending=[False, True])
    selected = ranking.iloc[0]
    chosen_id = str(selected.policy_id)
    config = {"k": float(selected.k), "persist_min": int(selected.persist_min)}
    published_id = "k6_p15"
    chosen_reports, published_reports = all_reports[chosen_id], all_reports[published_id]
    pd.DataFrame(trials).to_csv(output / "grid.csv", index=False)
    ranking.to_csv(output / "ranking.csv", index=False)
    _write_json(output / "grid_reports.json", all_reports)
    _write_json(output / "selected_config.json", config)
    _write_json(output / "published_config.json", PUBLISHED_CONFIG)
    for label, policy_id in (("selected", chosen_id), ("published", published_id)):
        _write_json(output / f"{label}_reports.json", all_reports[policy_id])
        for name, predictions in predictions_by_policy[policy_id].items():
            predictions.to_csv(output / f"{label}_{name}_predictions.csv", index=False)
    comparison = []
    for label, reports in (("B1_published", published_reports), ("B1_development_selected", chosen_reports)):
        for name, report in reports.items():
            comparison.append({"baseline": label, "dataset": name, **{key: report[key] for key in METRICS}})
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    published_reference = 78.01253946421124
    reproduced = published_reports["original_validation"]["composite_score"]
    selection = {
        "selected_config": config,
        "selected_policy_id": chosen_id,
        "selection_mean": float(selected.selection_mean),
        "objective": "Equal mean of composite score across familiar original validation and synthetic development",
        "datasets": {name: prepared[-1] for name, prepared in datasets.items()},
        "grid": {"k": list(K_GRID), "persist_min": list(PERSIST_GRID)},
        "tie_break": "First position in the declared k-major grid",
        "published_config": PUBLISHED_CONFIG,
        "published_reference_validation_score": published_reference,
        "published_reference_reproduced": abs(reproduced-published_reference) < 1e-8,
        "published_recomputed_validation_score": reproduced,
        "selected_reports": chosen_reports,
        "published_reports": published_reports,
        "implementation": "Unmodified retrofit_sim.baselines.regime_frame, zscores, b1_threshold and retrofit_sim.scoring.score",
        "causality_caveat": "Published B1 uses the complete first 24 hours for all early normalization; this early lookahead is retained, not silently repaired.",
        "source_sha256": {p.name: _sha(p) for p in [Path(B.__file__), SOURCE / "src/retrofit_sim/scoring.py", Path(__file__)]},
        "elapsed_seconds": time.perf_counter()-started,
        "holdout_evaluated": False,
    }
    _write_json(output / "selection.json", selection)
    print("Selected", config, "equal development mean", round(float(selected.selection_mean), 4), flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data", type=Path, default=SOURCE / "data/full")
    parser.add_argument("--development", type=Path, default=HERE / "synthetic_data/development")
    parser.add_argument("--output", type=Path, default=HERE / "baseline_results")
    args = parser.parse_args()
    select_b1(args.original_data, args.development, args.output)
