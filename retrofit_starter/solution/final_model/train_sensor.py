"""Reproduce the routed sensor expert using the frozen hist_01 regressors.

Run from this directory (or supply this script's path from another directory).
The five default datasets and their order match sensor_expert_03. An existing
run is never overwritten. Evaluation is optional and never changes the model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

from paths import HERE
from sensor_expert import COHERENCE, SENSOR_TYPES, SensorExpert

TRAIN_DATASETS = ("original_train", "train", "train_balanced", "train_crossed",
                  "train_sensor_crossed")
EVALUATION_DATASETS = ("development", "original_validation")
RETAIN_PREFIXES = ("level__", "std__", "range__", "fresh__", "age__", "count__",
                   "flat_minutes__", "equal_updates__", "residual__")
RETAIN_EXACT = set(COHERENCE.values()) | {
    "obs__steady", "obs__running", "obs__gap", "ctx__duty_demand_pct",
    "ctx__supply_water_temp_c", "ctx__ambient_temp_c", "phys__power_pu",
    "phys__electrical_power_factor",
}


def run_name(value):
    if value in ("", ".", "..") or Path(value).name != value:
        raise argparse.ArgumentTypeError("Use one run-directory name, without a path")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, type=run_name,
                        help="New directory below runs/; existing runs are refused")
    parser.add_argument("--reference", default="hist_01", type=run_name,
                        help="Frozen healthy-regressor model in runs/ (default: hist_01)")
    parser.add_argument("--datasets", nargs="+", choices=TRAIN_DATASETS,
                        default=list(TRAIN_DATASETS), help="Training datasets in sample order")
    parser.add_argument("--evaluate", nargs="*", choices=EVALUATION_DATASETS,
                        default=list(EVALUATION_DATASETS),
                        help="Evaluation datasets; pass this flag alone to skip evaluation")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--max-iter", type=int, default=140)
    parser.add_argument("--max-positive-per-type-channel", type=int, default=1800)
    parser.add_argument("--max-negative-per-channel", type=int, default=3500)
    return parser


def expert_parameters(args):
    return {"seed": args.seed, "max_positive_per_type_channel": args.max_positive_per_type_channel,
            "max_negative_per_channel": args.max_negative_per_channel,
            "max_iter": args.max_iter, "routed": True}


def retained_columns(frame):
    """Keep the original input-column order, including every peer residual."""
    return [name for name in frame if name.startswith(RETAIN_PREFIXES) or name in RETAIN_EXACT]


def training_command(args):
    """Portable, fully explicit command; execute with this directory as cwd."""
    command = ["python", "train_sensor.py", "--name", args.name,
               "--reference", args.reference, "--datasets", *args.datasets,
               "--evaluate", *args.evaluate, "--seed", str(args.seed),
               "--max-iter", str(args.max_iter),
               "--max-positive-per-type-channel", str(args.max_positive_per_type_channel),
               "--max-negative-per-channel", str(args.max_negative_per_channel)]
    return "OMP_NUM_THREADS=3 LOKY_MAX_CPU_COUNT=3 " + shlex.join(command)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def evaluate(expert, reference, dataset, output, prepared):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    from learning import make_targets

    meta, x, _, events, _ = prepared(dataset)
    x = reference.transform(x, meta)
    probabilities = expert.predict(x, meta)
    probabilities.to_parquet(output / f"{dataset}_probabilities.parquet", index=False)
    targets, _, _ = make_targets(meta, events)
    metrics = {}
    for fault in SENSOR_TYPES:
        known = targets[fault].ge(0)
        y = targets.loc[known, fault]
        p = probabilities.loc[known, fault]
        positive, negative = p[y.eq(1)], p[y.eq(0)]
        metrics[fault] = {
            "positive_minutes": int(y.eq(1).sum()),
            "average_precision": float(average_precision_score(y, p)) if len(positive) else None,
            "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
            "positive_median": float(positive.median()) if len(positive) else None,
            "negative_p99": float(negative.quantile(.99)) if len(negative) else None,
            "negative_above.8": int(negative.ge(.8).sum()),
        }
    behavior = []
    times = pd.to_datetime(meta.available_utc, utc=True)
    for _, event in events.iterrows():
        if event.fault_type not in SENSOR_TYPES or str(event.observable).lower() not in ("true", "1"):
            continue
        onset = pd.to_datetime(event.onset_utc, utc=True)
        detectable = pd.to_datetime(event.detectable_utc, utc=True)
        end = pd.to_datetime(event.end_utc, utc=True)
        start = max(onset, detectable) if pd.notna(detectable) else onset
        mask = meta.asset_id.eq(event.asset_id) & times.ge(start)
        if pd.notna(end):
            mask &= times.lt(end)
        p = probabilities.loc[mask, event.fault_type].to_numpy()
        if len(p):
            behavior.append({"dataset": dataset, "asset_id": str(event.asset_id),
                             "fault_type": event.fault_type, "sensor_tag": str(event.sensor_tag),
                             "minutes": len(p), "median": float(np.median(p)),
                             "p95": float(np.quantile(p, .95)),
                             "fraction_above.8": float(np.mean(p >= .8))})
    return metrics, behavior


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = HERE / "runs" / args.name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    reference_path = HERE / "runs" / args.reference / "model.joblib"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Frozen reference model is missing: {reference_path}")
    # Reserve the directory before loading any data; a concurrent invocation
    # cannot overwrite this run either. A failed run is preserved for diagnosis.
    output.mkdir(parents=True, exist_ok=False)

    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from run import prepared

    started = time.monotonic()
    def log(message):
        print(message, flush=True)
        with (output / "training.log").open("a") as handle:
            handle.write(message + "\n")

    parameters = expert_parameters(args)
    metadata = {
        "training_command": training_command(args),
        "command_working_directory": "directory containing train_sensor.py",
        "invocation_argv": [sys.executable, str(Path(__file__).resolve()),
                            *(sys.argv[1:] if argv is None else argv)],
        "datasets": args.datasets, "reference_model": args.reference,
        "frozen_reference_sha256": sha(reference_path), "parameters": parameters,
        "source_hashes": {name: sha(HERE / name) for name in
                          ("sensor_expert.py", "learning.py", "features.py",
                           "physics_features.py", "train_sensor.py", "run.py", "paths.py")},
        "versions": {"python": sys.version, "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "sample_order": "Datasets in listed order; prepared row and column order retained; "
                        "pd.concat(ignore_index=True); SensorExpert sorts channel names, "
                        "uses one seeded generator, then sorts retained row indices per channel.",
        "column_retention": {"prefixes": list(RETAIN_PREFIXES), "exact": sorted(RETAIN_EXACT)},
    }
    write_json(output / "reproduction.json", metadata)
    reference = joblib.load(reference_path)
    parts = [prepared(name) for name in args.datasets]
    meta = pd.concat([part[0] for part in parts], ignore_index=True)
    x = pd.concat([part[1] for part in parts], ignore_index=True)
    events = pd.concat([part[3] for part in parts], ignore_index=True)
    del parts
    gc.collect()
    x = reference.transform(x, meta)
    keep = retained_columns(x)
    x = x[keep].copy()
    gc.collect()
    metadata.update({"input_rows": len(x), "input_assets": int(meta.asset_id.nunique()),
                     "retained_columns": keep, "retained_column_count": len(keep)})
    log(f"Training {args.name}: {len(x)} rows, {metadata['input_assets']} assets, {len(keep)} retained columns")
    expert = SensorExpert(**parameters).fit(x, meta, events)
    joblib.dump(expert, output / "model.joblib", compress=3)
    metadata.update({"training": expert.training_info,
                     "model_sha256": sha(output / "model.joblib")})
    write_json(output / "reproduction.json", metadata)
    analysis = {**metadata, "metrics": {}, "event_behavior": []}
    del x, meta, events
    gc.collect()
    for dataset in args.evaluate:
        metrics, behavior = evaluate(expert, reference, dataset, output, prepared)
        analysis["metrics"][dataset] = metrics
        analysis["event_behavior"].extend(behavior)
        log(f"Evaluated {dataset}")
    analysis["seconds"] = time.monotonic() - started
    write_json(output / "analysis.json", analysis)
    log(f"Saved {output}; elapsed {analysis['seconds']:.1f}s")
    return output


if __name__ == "__main__":
    main()
