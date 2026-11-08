"""One-shot synthetic holdout evaluation after an explicit SHA256 freeze.

No holdout is generated here. The caller supplies the already frozen manifest's
trusted SHA256 separately. Every model/policy/source hash is checked before any
holdout file is opened; the generated holdout must attest to that same manifest.

Freeze schema (paths relative to this script's directory unless absolute):
  {"run": "chosen_run", "policy_dir": "runs/chosen_run/policy_tuning",
   "model_path": "runs/chosen_run/model.joblib",
   "policy_path": "runs/chosen_run/policy_tuning/policy.json",
   "b1_config_path": "baseline_results/selected_config.json",
   "frozen_at_utc": "2026-11-07T12:00:00Z",
   "artifacts": {"features.py": "<sha256>", "...": "<sha256>"}}

Artifacts must cover every root Python module, final model/policy, selected B1
configuration, and original baseline/scorer sources. Results keep published and
development-selected B1 distinct. An existing output directory cannot be reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time


from paths import HERE, SOURCE
METRICS = ("composite_score", "event_f1", "event_recall", "alert_precision", "timeliness",
           "fault_type_recall", "class_recall", "false_alerts_per_asset_day", "events_scored",
           "true_positive_events", "false_alert_episodes", "excess_repeat_episodes", "alerts_scored",
           "median_lead_time_h")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else HERE/path).resolve()


def verify_freeze(run, policy_dir, freeze_manifest, freeze_sha256):
    """Verify the immutable inputs without touching any holdout path."""
    freeze_manifest = resolve(freeze_manifest)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", freeze_sha256):
        raise ValueError("Supply the previously recorded 64-digit freeze SHA256")
    if sha256(freeze_manifest) != freeze_sha256.lower():
        raise ValueError("Freeze manifest differs from the separately supplied trusted SHA256")
    manifest = json.loads(freeze_manifest.read_text())
    for key in ("run", "policy_dir", "model_path", "policy_path", "b1_config_path", "frozen_at_utc", "artifacts"):
        if key not in manifest:
            raise ValueError(f"Freeze manifest lacks {key}")
    if manifest["run"] != run or resolve(manifest["policy_dir"]) != resolve(policy_dir):
        raise ValueError("Requested run/policy directory does not match the freeze")
    if resolve(manifest["model_path"]) != (HERE/"runs"/run/"model.joblib").resolve():
        raise ValueError("Frozen model must belong to the requested run")
    if resolve(manifest["policy_path"]) != resolve(policy_dir)/"policy.json":
        raise ValueError("Frozen policy must be policy.json in the requested policy directory")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Freeze artifacts must be a non-empty path-to-SHA256 mapping")
    resolved_hashes = {}
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"Invalid artifact SHA256: {name}")
        path = resolve(name)
        if path in resolved_hashes:
            raise ValueError(f"Duplicate resolved artifact: {path}")
        if not path.is_file() or sha256(path) != expected.lower():
            raise ValueError(f"Frozen artifact missing or changed: {path}")
        resolved_hashes[path] = expected.lower()
    required = {p.resolve() for p in HERE.glob("*.py")}
    required.update(resolve(manifest[key]) for key in ("model_path", "policy_path", "b1_config_path"))
    required.update((SOURCE/"src/retrofit_sim"/name).resolve() for name in ("baselines.py", "scoring.py"))
    missing = required-set(resolved_hashes)
    if missing:
        raise ValueError("Freeze does not cover required artifacts: " + ", ".join(str(p) for p in sorted(missing)))
    # Parse the declaration after verification. No inferred wall-clock timestamp
    # substitutes for an explicit immutable model/policy freeze.
    from datetime import datetime
    declared = datetime.fromisoformat(manifest["frozen_at_utc"].replace("Z", "+00:00"))
    if declared.tzinfo is None:
        raise ValueError("frozen_at_utc must include its UTC offset")
    return manifest, freeze_manifest


def _json_clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_json_clean(value), indent=2, allow_nan=False)+"\n")


def evaluate(run, policy_dir, freeze_manifest, freeze_sha256, data_dir=None, output=None):
    manifest, freeze_path = verify_freeze(run, policy_dir, freeze_manifest, freeze_sha256)
    data_dir = resolve(data_dir or "synthetic_data/holdout")
    output = resolve(output or f"runs/{run}/frozen_holdout")
    if output.exists():
        raise FileExistsError("A frozen evaluation is one shot; preserve the existing output directory")
    # These imports are intentionally after the artifact guard. Inference does
    # not load labels; labels below are supplied solely to the official scorer.
    import joblib
    import numpy as np
    import pandas as pd
    from alerts import AlertPolicy, make_alerts
    from features import build_features
    from baseline_compare import evaluate_b1, PUBLISHED_CONFIG
    sys.path.insert(0, str(SOURCE/"src"))
    from retrofit_sim.scoring import score

    holdout_manifest_sha256 = sha256(data_dir/"manifest.json")
    generated = json.loads((data_dir/"manifest.json").read_text())
    if generated.get("role") != "holdout":
        raise ValueError("Expected the explicitly generated synthetic holdout")
    if generated.get("frozen_pipeline_manifest_sha256") != freeze_sha256.lower():
        raise ValueError("Holdout was not generated against this exact frozen pipeline manifest")
    generated_artifacts = generated.get("artifacts", {})
    required_data = ("assets.csv", "features_1min.parquet", "events.csv")
    if any(name not in generated_artifacts for name in required_data):
        raise ValueError("Holdout provenance lacks required artifact hashes")
    for name in ("operating_context.csv", "vibration_snapshots.parquet", "benign_windows.csv"):
        if (data_dir/name).exists() and name not in generated_artifacts:
            raise ValueError(f"Holdout provenance omits an inference/scoring input: {name}")
    for name, expected in generated_artifacts.items():
        path = (data_dir/name).resolve()
        if path.parent != data_dir:
            raise ValueError("Holdout artifact paths must be local filenames")
        if sha256(path) != expected:
            raise ValueError(f"Holdout artifact changed: {name}")
    assets = pd.read_csv(data_dir/"assets.csv")
    if assets.empty or set(assets["split"]) != {"holdout"} or assets.asset_id.str.match(r"PUMP-\d+$").any():
        raise ValueError("Expected synthetic holdout assets only")
    if assets.asset_id.duplicated().any():
        raise ValueError("Duplicate holdout asset IDs")
    ids = assets.asset_id.tolist()
    observations = pd.read_parquet(data_dir/"features_1min.parquet", filters=[("asset_id", "in", ids)])
    context_path, shot_path = data_dir/"operating_context.csv", data_dir/"vibration_snapshots.parquet"
    context = pd.read_csv(context_path) if context_path.exists() else None
    snapshots = pd.read_parquet(shot_path, filters=[("asset_id", "in", ids)]) if shot_path.exists() else None
    events = pd.read_csv(data_dir/"events.csv")
    if set(events.asset_id)-set(ids):
        raise ValueError("Labels contain out-of-scope assets")
    benign_path = data_dir/"benign_windows.csv"
    benign = pd.read_csv(benign_path) if benign_path.exists() else None
    policy = AlertPolicy(**json.loads(resolve(manifest["policy_path"]).read_text()))
    b1_config = json.loads(resolve(manifest["b1_config_path"]).read_text())
    model = joblib.load(resolve(manifest["model_path"]))
    # Reserve the result directory before scoring, so accidental retries cannot
    # quietly replace an earlier look at holdout performance.
    output.mkdir(parents=True, exist_ok=False)
    _write(output/"evaluation_started.json", {"run": run, "freeze_sha256": freeze_sha256.lower(),
           "freeze_manifest": str(freeze_path), "data_dir": str(data_dir),
           "holdout_manifest_sha256": holdout_manifest_sha256})
    started = time.perf_counter()
    # Use the explicit independent end clock, including terminal silence. Both
    # base and composite learners expose predict_proba(x, meta).
    own_parts = []
    for _, asset in assets.iterrows():
        meta, x = build_features(observations, asset, context, snapshots, until=asset.window_end_utc)
        probability = model.predict_proba(x, meta)
        pred = make_alerts(meta, probability, policy)
        if len(pred):
            pred = pred.loc[pd.to_datetime(pred.alert_time_utc, utc=True).lt(pd.Timestamp(asset.window_end_utc))]
        own_parts.append(pred)
    own = pd.concat(own_parts, ignore_index=True)
    predictions = {"Quaviz_frozen": own}
    reports = {"Quaviz_frozen": score(own, events, assets, benign)}
    # Also retain the separate familiar-validation optimum from the declared
    # 20-policy B1 grid. All three configurations are fixed before holdout.
    baseline_configs={"B1_published": PUBLISHED_CONFIG,
        "B1_development_selected": b1_config,
        "B1_validation_selected": {"k":6.0,"persist_min":5}}
    for label, config in baseline_configs.items():
        predictions[label], reports[label] = evaluate_b1(data_dir, config)
    # Detect code/model/policy mutation during the run before releasing results.
    verify_freeze(run, policy_dir, freeze_path, freeze_sha256)
    if sha256(data_dir/"manifest.json") != holdout_manifest_sha256:
        raise ValueError("Holdout manifest changed during evaluation")
    for name, expected in generated_artifacts.items():
        if sha256(data_dir/name) != expected:
            raise ValueError(f"Holdout input changed during evaluation: {name}")
    rows, per_asset = [], []
    for label, pred in predictions.items():
        pred.to_csv(output/f"{label}_predictions.csv", index=False)
        _write(output/f"{label}_report.json", reports[label])
        rows.append({"model": label, **{k: reports[label][k] for k in METRICS}})
        for _, asset in assets.iterrows():
            scoped = assets.loc[assets.asset_id.eq(asset.asset_id)]
            r = score(pred, events, scoped, benign)
            per_asset.append({"model": label, "asset_id": asset.asset_id,
                              "profile": asset.profile, "instrumentation_tier": asset.instrumentation_tier,
                              **{k: r[k] for k in METRICS}})
    pd.DataFrame(rows).to_csv(output/"comparison.csv", index=False)
    paired = pd.DataFrame(per_asset)
    paired.to_csv(output/"per_asset.csv", index=False)
    pivot = paired.pivot(index="asset_id", columns="model", values="composite_score")
    deltas = {label: (pivot["Quaviz_frozen"]-pivot[label]).to_dict()
              for label in baseline_configs}
    summary = {"run": run, "freeze_sha256": freeze_sha256.lower(), "holdout_only": True,
               "reports": reports, "paired_asset_score_deltas": deltas,
               "asset_count": len(assets), "elapsed_seconds": time.perf_counter()-started,
               "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
               "interpretation": "One frozen public-simulator holdout. It does not establish field or private-test performance.",
               "baseline_caveat": "Unmodified published B1 retains first-day normalization lookahead; fixed and development-selected parameters are separate."}
    _write(output/"evaluation.json", summary)
    print(pd.DataFrame(rows)[["model", "composite_score", "event_recall", "fault_type_recall", "alert_precision"]].to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--data-dir", default="synthetic_data/holdout")
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluate(args.run, args.policy_dir, args.freeze_manifest, args.freeze_sha256, args.data_dir, args.output)
