"""Does a healthy-only autoencoder help the final model's classifier?

Every arm trains the same unrouted boosted-tree view (the hist_01 recipe) on the
seven official training pumps and is scored on official validation (PUMP-08/09).
Arms differ only in the normal-operation evidence given to the classifier:

  none         no normal model
  regressors   the final model's 14 per-channel healthy regressors (residuals)
  autoencoder  leave-one-channel-out errors of a fleet-wide denoising autoencoder
  both         regressor residuals and autoencoder errors

The autoencoder is fitted on healthy steady training minutes only, with no labels.
At time t it uses only the readings of minute t, and the trailing features use only
earlier minutes, so alerts stay causal.

  python ablation.py
  python ablation.py --seed 1 --arms regressors autoencoder
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[1] / "final_model"
sys.path.insert(0, str(FINAL))
from paths import SOURCE  # noqa: E402

sys.path.insert(0, str(SOURCE / "src"))
from retrofit_sim.scoring import score  # noqa: E402
from alerts import AlertPolicy, make_alerts  # noqa: E402
from features import CHANNELS  # noqa: E402
from learning import FAULTS, LearnedModel, make_targets  # noqa: E402
from run import prepared  # noqa: E402

OUT = HERE / "results"
SEED = 20261107
CONTEXT_PREFIXES = ("ctx__", "rating__", "control__", "profile__")
METRICS = ["composite_score", "event_f1", "timeliness", "fault_type_recall", "class_recall",
           "false_alerts_per_asset_day", "true_positive_events", "events_scored"]


class Net(torch.nn.Module):
    def __init__(self, n_channels, n_context, latent=4, hidden=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2 * n_channels + n_context, hidden), torch.nn.GELU(),
                                           torch.nn.Linear(hidden, latent))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(latent + n_context, hidden), torch.nn.GELU(),
                                           torch.nn.Linear(hidden, n_channels))

    def forward(self, values, mask, context):
        z = self.encoder(torch.cat([values * mask, mask, context], 1))
        return self.decoder(torch.cat([z, context], 1))


class HealthyAutoencoder:
    """Conditional denoising autoencoder of the 14 rating-scaled channel levels."""

    def fit(self, x, healthy_steady, seed=SEED, epochs=40, drop=0.25):
        torch.manual_seed(seed)
        self.channels = [c for c in CHANNELS if self._observed(x, c)[healthy_steady].sum() >= 80]
        self.context = [c for c in x if c.startswith(CONTEXT_PREFIXES)] + ["obs__running", "obs__run_minutes"]
        values, mask = self._channels(x, fit_rows=healthy_steady)
        ctx = self._context(x, fit_rows=healthy_steady)
        rows = np.flatnonzero(healthy_steady)
        v, m, c = (torch.tensor(a[rows]) for a in (values, mask, ctx))
        self.net = Net(len(self.channels), len(self.context))
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3, weight_decay=1e-5)
        for _ in range(epochs):
            for batch in torch.randperm(len(rows)).split(512):
                keep = m[batch] * (torch.rand_like(m[batch]) > drop).float()
                recon = self.net(v[batch], keep, c[batch])
                loss = (((recon - v[batch]) ** 2) * m[batch]).sum() / m[batch].sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        errors = self.errors(x)[rows]
        med = np.nanmedian(errors, 0)
        self.scale = np.maximum(np.nanmedian(np.abs(errors - med), 0) * 1.4826, 0.003)
        return self

    def _observed(self, x, c):
        return (x["level__" + c].notna() & x["fresh__" + c].gt(0)).to_numpy()

    def _channels(self, x, fit_rows=None):
        raw = np.column_stack([x["level__" + c].to_numpy(float) for c in self.channels])
        mask = np.column_stack([self._observed(x, c) for c in self.channels]).astype(np.float32)
        if fit_rows is not None:
            ok = fit_rows[:, None] & (mask > 0)
            self.center = np.array([np.median(raw[ok[:, j], j]) for j in range(raw.shape[1])])
            iqr = np.array([np.subtract(*np.percentile(raw[ok[:, j], j], [75, 25])) for j in range(raw.shape[1])])
            self.spread = np.maximum(iqr / 1.349, 1e-3)
        values = np.where(mask > 0, (raw - self.center) / self.spread, 0.0)
        return values.astype(np.float32), mask

    def _context(self, x, fit_rows=None):
        raw = x[self.context].to_numpy(float)
        if fit_rows is not None:
            self.ctx_mean = np.nanmean(raw[fit_rows], 0)
            self.ctx_std = np.where(np.nanstd(raw[fit_rows], 0) > 0, np.nanstd(raw[fit_rows], 0), 1.0)
        return np.nan_to_num((raw - self.ctx_mean) / self.ctx_std).astype(np.float32)

    @torch.no_grad()
    def errors(self, x):
        """Signed error of each channel when it is hidden and rebuilt from the others."""
        values, mask = self._channels(x)
        v, m, c = torch.tensor(values), torch.tensor(mask), torch.tensor(self._context(x))
        out = np.full(values.shape, np.nan, dtype=np.float32)
        for j in range(len(self.channels)):
            hidden = m.clone()
            hidden[:, j] = 0
            recon = self.net(v, hidden, c)[:, j].numpy()
            out[:, j] = np.where(mask[:, j] > 0, values[:, j] - recon, np.nan)
        return out

    def features(self, x, meta):
        err = np.clip(self.errors(x) / self.scale, -100, 100)
        added = {}
        groups = list(meta.groupby("asset_id", sort=False).indices.values())
        for j, ch in enumerate(self.channels):
            added["ae_resid__" + ch] = err[:, j]
            for window in [10, 60]:
                mean = np.full(len(x), np.nan, dtype=np.float32)
                delta = np.full(len(x), np.nan, dtype=np.float32)
                for ids in groups:
                    s = pd.Series(err[ids, j])
                    mean[ids] = s.rolling(window, min_periods=2).mean()
                    delta[ids] = s - s.shift(window)
                added[f"ae_mean{window}__" + ch] = mean
                added[f"ae_delta{window}__" + ch] = delta
        add = pd.DataFrame(added, index=x.index, dtype=np.float32)
        instrument = [f"ae_resid__{c}" for c in self.channels if not any(s in c for s in ["vibration", "speed", "ambient"])]
        absolute = add[instrument].abs()
        add["ae_sensor_max"] = absolute.max(axis=1)
        add["ae_sensor_mean"] = absolute.mean(axis=1)
        add["ae_sensor_dominance"] = absolute.max(axis=1) / (absolute.sum(axis=1) + 1e-6)
        return pd.concat([x, add], axis=1)


def policy_grid():
    for threshold, persistence, recovery in itertools.product([.55, .7, .8, .9, .96], [5, 15], [60, 240]):
        yield AlertPolicy(threshold=threshold, persistence=persistence, recovery=recovery)


def minute_metrics(probs, y):
    """Threshold-free: can the scores separate faulty from other minutes?"""
    out = {}
    for fault in FAULTS:
        keep = y[fault].ne(-1).to_numpy()
        labels = y[fault].to_numpy()[keep]
        if labels.min() == labels.max():
            continue
        out[fault] = roc_auc_score(labels, probs[fault].to_numpy()[keep])
    known = (y != -1).all(axis=1).to_numpy()
    any_fault = (y == 1).any(axis=1).to_numpy()[known]
    p_any = probs[FAULTS].max(axis=1).to_numpy()[known]
    return {"per_fault_auroc": out, "mean_fault_auroc": float(np.mean(list(out.values()))),
            "any_fault_ap": float(average_precision_score(any_fault, p_any)),
            "any_fault_auroc": float(roc_auc_score(any_fault, p_any))}


def main(seed=SEED, selected=None):
    out = OUT if seed == SEED else OUT / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    meta, x, _, events, _ = prepared("original_train")
    vmeta, vx, vassets, vevents, vbenign = prepared("original_validation")
    y, w, healthy = make_targets(meta, events)
    vy, _, _ = make_targets(vmeta, vevents)

    t0 = time.perf_counter()
    ae = HealthyAutoencoder().fit(x, healthy & meta.steady.to_numpy(), seed=seed)
    print(f"autoencoder: {len(ae.channels)} channels, fitted in {time.perf_counter() - t0:.0f} s", flush=True)
    x_ae, vx_ae = ae.features(x, meta), ae.features(vx, vmeta)

    arms = {"none": (False, x, vx), "regressors": (True, x, vx),
            "autoencoder": (False, x_ae, vx_ae), "both": (True, x_ae, vx_ae)}
    arms = {k: v for k, v in arms.items() if not selected or k in selected}
    rows, per_type, minute = [], [], {}
    for arm, (residuals, train_x, val_x) in arms.items():
        t0 = time.perf_counter()
        model = LearnedModel(kind="hist", include_residuals=residuals, seed=seed)
        if residuals:
            model.fit_reference(train_x, meta, healthy)
        model.fit_classifier(model.transform(train_x, meta), meta, y, w)
        probs = model.probabilities_from_transformed(model.transform(val_x, vmeta))
        minute[arm] = minute_metrics(probs, vy)
        best = None
        for policy in policy_grid():
            pred = make_alerts(vmeta, probs, policy)
            report = score(pred, vevents, vassets, vbenign)
            row = {"arm": arm, **asdict(policy), **{k: report[k] for k in METRICS}}
            rows.append(row)
            if best is None or row["composite_score"] > best[0]["composite_score"]:
                best = (row, report)
        for event in best[1].get("per_event", []):
            per_type.append({"arm": arm, **event})
        print(f"{arm:12s} best composite {best[0]['composite_score']:.1f}  mean fault AUROC "
              f"{minute[arm]['mean_fault_auroc']:.3f}  ({time.perf_counter() - t0:.0f} s)", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(out / "policy_grid.csv", index=False)
    if per_type:
        pd.DataFrame(per_type).to_csv(out / "per_event_best_policy.csv", index=False)
    (out / "minute_metrics.json").write_text(json.dumps(minute, indent=2) + "\n")
    best = grid.loc[grid.groupby("arm", sort=False).composite_score.idxmax()]
    common = grid.groupby(["threshold", "persistence", "recovery"]).composite_score.mean().idxmax()
    fixed = grid.set_index(["threshold", "persistence", "recovery"]).loc[common].reset_index()
    summary = pd.DataFrame({
        "best_policy_composite": best.set_index("arm").composite_score,
        "shared_policy_composite": fixed.set_index("arm").composite_score,
        "median_over_grid": grid.groupby("arm").composite_score.median(),
        "mean_fault_auroc": pd.Series({a: m["mean_fault_auroc"] for a, m in minute.items()}),
        "any_fault_ap": pd.Series({a: m["any_fault_ap"] for a, m in minute.items()}),
    }).loc[list(arms)].round(3)
    summary.to_csv(out / "summary.csv")
    print(f"\nshared policy = threshold {common[0]}, persistence {common[1]}, recovery {common[2]}")
    print(summary.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--arms", nargs="+", choices=["none", "regressors", "autoencoder", "both"])
    a = p.parse_args()
    main(a.seed, a.arms)
