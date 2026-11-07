"""Bounded supervised neural comparison using observed physical feature routes.

Training streams original_train/train/train_balanced/train_crossed, one at a
time. Each binary target masks unknown labels, balances positive events inside
each source, and balances positive/negative class loss. Normalization is fitted
only on sampled training rows. Independent small MLPs handle concurrent faults.
No validation data are used to fit neural weights, imputation or scaling.

python neural.py train --name neural_01 --hidden 32 16 --activation tanh
python neural.py predict --name neural_01
python neural.py assess --name neural_01

The assess phase retains a declared 16-policy development sweep plus the fixed
0.8/10min policy. It reads only familiar validation and synthetic development.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from alerts import AlertPolicy, make_alerts
from learning import FAULTS, LearnedModel, expert_columns, make_targets
from run import HERE, prepared
from retrofit_sim.scoring import score


TRAINING = ("original_train", "train", "train_balanced", "train_crossed")
DEVELOPMENT = ("original_validation", "development")
METRICS = ("composite_score", "event_f1", "event_recall", "alert_precision", "fault_type_recall",
           "class_recall", "timeliness", "events_scored", "true_positive_events", "false_alert_episodes",
           "excess_repeat_episodes", "alerts_scored")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(_clean(value), indent=2, allow_nan=False)+"\n")


@dataclass
class TrainingScale:
    columns: list
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    present: np.ndarray | None = None

    def fit(self, matrix):
        self.present = np.isfinite(matrix).any(axis=0)
        observed = matrix[:, self.present]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.center = np.nanmedian(observed, axis=0).astype(np.float32)
            quartiles = np.nanpercentile(observed, [25., 75.], axis=0)
            robust = (quartiles[1]-quartiles[0])/1.349
            fallback = np.nanstd(observed, axis=0)
        # Degenerate mostly-binary/rare signals retain scale instead of being
        # amplified arbitrarily. No evaluation statistics enter these values.
        self.scale = np.where(robust > 1e-4, robust, np.where(fallback > 1e-4, fallback, 1.)).astype(np.float32)
        return self

    def encode(self, matrix):
        matrix = matrix[:, self.present]
        missing = ~np.isfinite(matrix)
        scaled = (np.where(missing, self.center, matrix)-self.center)/self.scale
        return np.concatenate([np.clip(scaled, -8., 8.), missing.astype(np.float32)], axis=1).astype(np.float32)


@dataclass
class NeuralModel:
    reference: object | None = None
    classifiers: dict = field(default_factory=dict)
    scalers: dict = field(default_factory=dict)
    training_info: dict = field(default_factory=dict)

    def transform(self, x, meta):
        return self.reference.transform(x, meta) if self.reference is not None else x.copy()

    def probabilities_from_transformed(self, transformed):
        result = {}
        with threadpool_limits(limits=2):
            for fault in FAULTS:
                model = self.classifiers[fault]
                if isinstance(model, (int, float)):
                    result[fault] = np.full(len(transformed), model)
                elif not len(transformed):
                    result[fault] = np.empty(0)
                else:
                    scaler = self.scalers[fault]
                    matrix = transformed.reindex(columns=scaler.columns).to_numpy(dtype=np.float32)
                    result[fault] = model.predict_proba(scaler.encode(matrix))[:, 1]
        return pd.DataFrame(result, index=transformed.index)

    def predict_proba(self, x, meta):
        return self.probabilities_from_transformed(self.transform(x, meta))


def train(args):
    output = HERE/"runs"/args.name
    output.mkdir(parents=True, exist_ok=True)
    if (output/"training.json").exists() or (output/"model.joblib").exists():
        raise FileExistsError("Preserve earlier neural attempts; choose a new run name")
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    model = NeuralModel()
    reference_path = HERE/"runs"/args.reference_from/"model.joblib" if args.reference_from else None
    if reference_path is not None:
        prior = joblib.load(reference_path)
        model.reference = LearnedModel(regressors=prior.regressors, include_residuals=True, include_waveforms=True)
        del prior
    samples = {fault: [] for fault in FAULTS}
    labels = {fault: [] for fault in FAULTS}
    sources = []
    routes = None
    per_source_class = max(100, args.max_samples//(2*len(TRAINING)))
    # Only compact routed samples are retained across source datasets.
    for name in TRAINING:
        meta, x, assets, events, _ = prepared(name)
        y, event_weights, _ = make_targets(meta, events)
        transformed = model.transform(x, meta)
        if routes is None:
            routes = {f: expert_columns(f, [c for c in transformed if not c.startswith("value__")]) for f in FAULTS}
        source_info = {"dataset": name, "rows": len(x), "assets": len(assets), "samples": {}}
        for fault in FAULTS:
            positive = np.flatnonzero(y[fault].eq(1))
            negative = np.flatnonzero(y[fault].eq(0))
            if len(positive):
                mass = event_weights.iloc[positive][fault].to_numpy()
                positive = rng.choice(positive, per_source_class, replace=True, p=mass/mass.sum())
            if len(negative):
                negative = rng.choice(negative, min(len(negative), per_source_class), replace=False)
            selected = np.r_[positive, negative]
            if len(selected):
                samples[fault].append(transformed.iloc[selected][routes[fault]].to_numpy(dtype=np.float32))
                labels[fault].append(np.r_[np.ones(len(positive), np.int8), np.zeros(len(negative), np.int8)])
            source_info["samples"][fault] = {"positive": len(positive), "negative": len(negative)}
        sources.append(source_info)
        print("sampled", name, "rows", len(x), "seconds", round(time.perf_counter()-started, 1), flush=True)
        del meta, x, assets, events, transformed, y, event_weights
        gc.collect()
    settings = {"datasets": list(TRAINING), "hidden": list(args.hidden), "activation": args.activation,
                "max_iter": args.max_iter, "max_samples_per_fault": args.max_samples, "seed": args.seed,
                "reference_from": args.reference_from, "reference_sha256": sha(reference_path) if reference_path else None,
                "scaling": "Training-only median and IQR, std fallback; clip +/-8; append missingness indicators",
                "sampling": "Equal source budgets; inverse-duration event-balanced positive resampling; uniform negatives; class-balanced loss",
                "early_stopping": False, "internal_validation_split": None, "sources": sources,
                "source_sha256": {p.name: sha(p) for p in [HERE/n for n in ("neural.py", "learning.py", "features.py", "physics_features.py", "alerts.py")]}}
    write(output/"training_started.json", settings)
    for fault in FAULTS:
        if time.perf_counter()-started > args.max_seconds:
            write(output/"budget_stop.json", {"elapsed_seconds": time.perf_counter()-started,
                  "completed_faults": list(model.classifiers), "next_fault": fault})
            raise RuntimeError("Neural experiment reached its declared runtime budget before all classes completed")
        before = time.perf_counter()
        matrix = np.concatenate(samples.pop(fault), axis=0)
        target = np.concatenate(labels.pop(fault), axis=0)
        if len(np.unique(target)) < 2:
            model.classifiers[fault] = float(target[0]) if len(target) else 0.
            continue
        scaler = TrainingScale(routes[fault]).fit(matrix)
        encoded = scaler.encode(matrix)
        counts = np.bincount(target, minlength=2)
        weight = np.asarray([len(target)/(2.*counts[t]) for t in target], dtype=np.float32)
        estimator = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), activation=args.activation,
            solver="adam", alpha=.003, batch_size=256, learning_rate_init=.001,
            max_iter=args.max_iter, early_stopping=False, tol=1e-4, n_iter_no_change=8,
            shuffle=True, random_state=args.seed)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(encoded, target, sample_weight=weight)
        model.classifiers[fault], model.scalers[fault] = estimator, scaler
        info = {"rows": len(target), "positive": int(counts[1]), "negative": int(counts[0]),
                "routed_features": len(routes[fault]), "encoded_features": encoded.shape[1],
                "iterations": estimator.n_iter_, "final_training_loss": estimator.loss_,
                "training_loss_curve": estimator.loss_curve_, "seconds": time.perf_counter()-before,
                "warnings": [str(w.message) for w in caught]}
        model.training_info[fault] = info
        write(output/"training_progress.json", model.training_info)
        print("fit", fault, "epochs", estimator.n_iter_, "loss", round(estimator.loss_, 4),
              "seconds", round(info["seconds"], 1), flush=True)
        del matrix, target, encoded, weight
        gc.collect()
    joblib.dump(model, output/"model.joblib", compress=3)
    write(output/"training.json", {**settings, "faults": model.training_info,
          "elapsed_seconds": time.perf_counter()-started, "model_sha256": sha(output/"model.joblib")})
    print("saved", output/"model.joblib", "seconds", round(time.perf_counter()-started, 1), flush=True)


def predict(args):
    output = HERE/"runs"/args.name
    model = joblib.load(output/"model.joblib")
    for name in DEVELOPMENT:
        meta, x, _, _, _ = prepared(name)
        model.predict_proba(x, meta).to_parquet(output/f"{name}_probabilities.parquet", index=False)
        print("predicted", name, flush=True)
        del meta, x
        gc.collect()


def assess(args):
    output = HERE/"runs"/args.name
    if (output/"neural_development_selection.json").exists():
        raise FileExistsError("Preserve earlier development assessment")
    datasets = {name: prepared(name) for name in DEVELOPMENT}
    probabilities = {name: pd.read_parquet(output/f"{name}_probabilities.parquet") for name in DEVELOPMENT}
    rows, reports = [], {}
    configurations = [(t, p) for t, p in itertools.product((.2, .4, .6, .8), (3, 5, 10, 15))]
    for threshold, persistence in configurations:
        policy = AlertPolicy(threshold=threshold, persistence=persistence, recovery=60, reminder=1440)
        key = f"t{threshold}_p{persistence}"
        reports[key] = {}
        for name, (meta, _, assets, events, benign) in datasets.items():
            predictions = make_alerts(meta, probabilities[name], policy)
            report = score(predictions, events, assets, benign)
            reports[key][name] = report
            rows.append({"policy_id": key, "dataset": name, "threshold": threshold, "persistence": persistence,
                         **{metric: report[metric] for metric in METRICS}})
    table = pd.DataFrame(rows)
    ranking = table.groupby("policy_id").composite_score.mean().sort_values(ascending=False)
    selected = ranking.index[0]
    chosen = table.loc[table.policy_id.eq(selected)].iloc[0]
    policy = AlertPolicy(threshold=float(chosen.threshold), persistence=int(chosen.persistence), recovery=60, reminder=1440)
    table.to_csv(output/"neural_development_grid.csv", index=False)
    write(output/"neural_development_reports.json", reports)
    write(output/"neural_development_selection.json", {"policy_id": selected, "selection_mean": ranking.iloc[0],
          "reports": reports[selected], "fixed_0.8_10_reports": reports["t0.8_p10"],
          "development_only": True, "holdout_evaluated": False})
    from dataclasses import asdict
    write(output/"policy.json", asdict(policy))
    for name, (meta, _, _, _, _) in datasets.items():
        make_alerts(meta, probabilities[name], policy).to_csv(output/f"{name}_alerts.csv", index=False)
    print("selected", selected, "equal_mean", ranking.iloc[0],
          {d: r["composite_score"] for d, r in reports[selected].items()}, flush=True)
    print("fixed_0.8_10", {d: r["composite_score"] for d, r in reports["t0.8_p10"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "predict", "assess"))
    parser.add_argument("--name", default="neural_01")
    parser.add_argument("--hidden", type=int, nargs=2, default=[32, 16])
    parser.add_argument("--activation", choices=("tanh", "relu"), default="tanh")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--max-seconds", type=float, default=480)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--reference-from", default="hist_01")
    args = parser.parse_args()
    {"train": train, "predict": predict, "assess": assess}[args.phase](args)


if __name__ == "__main__":
    # Keep pickled class references importable as neural.NeuralModel.
    from neural import main as entrypoint
    entrypoint()
