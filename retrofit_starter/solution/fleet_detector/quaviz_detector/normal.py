"""Stage 1: fleet-wide normal-behavior model.

For every running minute it predicts what each channel should read on a healthy pump and how far off a healthy
pump typically is, giving a standardized deviation per channel:

  model A  operating point: channel from the operating context only (duty, temperatures, sea state, tank level,
           schedule, time since start) and static nameplate data. The controller's outputs (speed, valve position)
           are targets here, so a controller compensating for a fault shows up as a deviation.
  model B  consistency: channel from the same inputs plus the measured speed, valve position and fluid
           temperature, i.e. how the channels normally move together (the pump and system curve, learned from data).

Values are nameplate-scaled (``features.minute_frame``), so one model serves pumps of every size. Inputs never
include a channel's own recent history, so slow faults are not absorbed into the prediction. The spread model is
fitted on residuals from pumps the mean model did not see, so a deviation of 3 means "three times the usual error on
a new pump".
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CONTROLS, PROFILES, TIERS

TARGETS_A = ["n", "vp", "i", "pw", "pd", "ps", "q", "tf", "bt", "wt", "v", "a", "c"]
TARGETS_B = ["i", "pw", "pd", "ps", "q", "bt", "wt", "v", "a", "c"]
MEASURED_INPUTS = ["n", "vp", "tf"]
CONTEXT_INPUTS = ["duty", "supply_t", "sea_t", "hs", "level", "amb_ctx"]


def static_inputs(a: pd.Series) -> dict:
    """Nameplate ratios and one-hot type; no pump-curve formulas."""
    f = lambda k: float(a[k]) if pd.notna(a.get(k)) else np.nan
    h, q = f("pump_rated_head_m"), f("pump_rated_flow_m3_h")
    shaft = 9.81 * q / 3600 * h / f("pump_bep_efficiency")
    d = {"shutoff_ratio": f("pump_shutoff_head_m") / h, "design_q_ratio": f("design_flow_m3_h") / q,
         "design_h_ratio": f("design_head_m") / h, "static_h_ratio": f("static_head_m") / h,
         "npsh_ratio": f("npsh3_at_bep_m") / h, "load_ratio": shaft / f("motor_rated_kw"), "eta_bep": f("pump_bep_efficiency"),
         "rated_krpm": f("motor_rated_rpm") / 1000, "supply_hz": f("supply_frequency_hz"), "min_speed_ratio": f("min_speed_ratio"),
         "log_q": np.log(q), "log_h": np.log(h), "vib_alarm": f("vibration_alarm_mm_s")}
    d |= {f"profile_{p}": float(a["profile"] == p) for p in PROFILES}
    d |= {f"control_{c}": float(a["control"] == c) for c in CONTROLS}
    d |= {f"tier_{t}": float(a["instrumentation_tier"] == t) for t in TIERS}
    return d


class _Fit:
    """Gradient-boosted regressor on the non-constant input columns (scikit-learn's weighted binning fails on a
    constant column)."""

    def __init__(self, X, y, w, max_iter: int, seed: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns (channel absent in this subset)
            lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        self.cols = np.flatnonzero(np.isfinite(lo) & (hi > lo))
        self.m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                                               l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                               n_iter_no_change=20, random_state=seed).fit(X[:, self.cols], y, sample_weight=w)

    def predict(self, X):
        return self.m.predict(X[:, self.cols])


class NormalModel:
    def __init__(self, max_iter: int = 300, seed: int = 0):
        self.max_iter, self.seed = max_iter, seed
        self.schedules: list[str] = []
        self.mean: dict = {}
        self.spread: dict = {}
        self.floor: dict = {}

    # ---- inputs -------------------------------------------------------------------------------------------------
    def _context(self, fr: pd.DataFrame, a: pd.Series) -> pd.DataFrame:
        X = pd.DataFrame(index=fr.index)
        for k in CONTEXT_INPUTS:
            X[k] = fr[k].astype(float) if k in fr else np.nan
        age = fr["run_age_min"].astype(float)
        X["run_age_h"] = np.minimum(age, 1440.0) / 60.0
        X["log_run_age"] = np.log1p(age)
        sched = fr["schedule"].astype(str)
        for s in self.schedules:
            X[f"sched_{s}"] = (sched == s).astype(float)
        for k, v in static_inputs(a).items():
            X[k] = v
        return X

    def _inputs(self, fr: pd.DataFrame, a: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        XA = self._context(fr, a)
        XB = XA.copy()
        for k in MEASURED_INPUTS:
            XB[f"in_{k}"] = fr[k].astype(float) if k in fr else np.nan
        return XA, XB

    # ---- fit ----------------------------------------------------------------------------------------------------
    def fit(self, samples: list[tuple[pd.DataFrame, pd.Series, np.ndarray, int, float]]) -> "NormalModel":
        """``samples``: (minute frame, asset row, healthy mask, keep every n-th healthy minute, sample weight) per pump."""
        self.schedules = sorted({s for fr, *_ in samples for s in fr["schedule"].dropna().astype(str).unique()})
        XA, XB, Y, grp, wt = [], [], [], [], []
        for g, (fr, a, mask, every, weight) in enumerate(samples):
            idx = np.flatnonzero(mask)[::every]
            if not len(idx):
                continue
            xa, xb = self._inputs(fr.iloc[idx], a)
            XA.append(xa)
            XB.append(xb)
            Y.append(pd.DataFrame({k: fr[k].iloc[idx].astype(float).values if k in fr else np.nan for k in TARGETS_A},
                                  index=xa.index))
            grp.append(np.full(len(idx), g))
            wt.append(np.full(len(idx), float(weight)))
        XA, XB, Y = pd.concat(XA), pd.concat(XB), pd.concat(Y)
        grp, wt = np.concatenate(grp), np.concatenate(wt)
        self.cols = {"A": list(XA.columns), "B": list(XB.columns)}
        half = (grp % 2).astype(bool)
        for model, X, targets in (("A", XA, TARGETS_A), ("B", XB, TARGETS_B)):
            Xv = X.to_numpy(np.float32)
            for k in targets:
                y = Y[k].to_numpy(float)
                ok = np.isfinite(y)
                if ok.sum() < 2000:
                    continue
                # out-of-pump residuals: each half of the pumps predicted by a model fitted on the other half
                r = np.full(len(y), np.nan)
                for side in (False, True):
                    tr, te = ok & (half != side), ok & (half == side)
                    if tr.sum() >= 1000 and te.any():
                        r[te] = y[te] - _Fit(Xv[tr], y[tr], wt[tr], self.max_iter, self.seed).predict(Xv[te])
                self.mean[(model, k)] = _Fit(Xv[ok], y[ok], wt[ok], self.max_iter, self.seed)
                rr = np.isfinite(r)
                ar = np.abs(r[rr])
                self.spread[(model, k)] = _Fit(Xv[rr], ar, wt[rr], self.max_iter // 2, self.seed)
                self.floor[(model, k)] = 0.25 * 1.2533 * float(np.median(ar))
        return self

    # ---- predict ------------------------------------------------------------------------------------------------
    def deviations(self, fr: pd.DataFrame, a: pd.Series) -> pd.DataFrame:
        """Standardized deviation ``z<A|B>_<channel>`` per minute; NaN while stopped or where the channel is missing."""
        out = pd.DataFrame(index=fr.index)
        run = fr["running"].to_numpy(bool)
        XA, XB = self._inputs(fr, a)
        X = {"A": XA.reindex(columns=self.cols["A"]).to_numpy(np.float32),
             "B": XB.reindex(columns=self.cols["B"]).to_numpy(np.float32)}
        for (model, k), m in self.mean.items():
            col = f"z{model}_{k}"
            if k not in fr or not run.any():
                out[col] = np.nan
                continue
            z = np.full(len(fr), np.nan)
            xs = X[model][run]
            sd = np.maximum(1.2533 * self.spread[(model, k)].predict(xs), self.floor[(model, k)])
            z[run] = (fr[k].to_numpy(float)[run] - m.predict(xs)) / sd
            out[col] = z
        return out
