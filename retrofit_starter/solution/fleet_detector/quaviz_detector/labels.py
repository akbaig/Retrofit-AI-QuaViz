"""Minute-level views of the event labels, used only for training (never as model inputs).

Windows follow the official scorer: an event lasts from its onset to ``end_utc`` (plus 30 min for events that end
without a repair: cavitation, air entrainment, instrument faults), or to the asset window end when open-ended.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset, ts

NO_REPAIR = {"CAVITATION", "AIR_ENTRAINMENT"}
POST_TOL = pd.Timedelta(minutes=30)


def asset_events(ds: Dataset, asset_id: str) -> pd.DataFrame:
    """Events of one asset with parsed times and the effective end used by the scorer."""
    e = ds.events[ds.events["asset_id"] == asset_id].copy()
    if not len(e):
        return e
    a = ds.asset(asset_id)
    w_end = ts(pd.Series([a["window_end_utc"]]))[0]
    for c in ("onset_utc", "detectable_utc", "action_deadline_utc", "end_utc"):
        e[c] = ts(e[c])
    e["observable"] = e["observable"].astype(str).str.lower().isin(["true", "1"])
    tol = np.where((e["event_class"] == "SENSOR") | e["fault_type"].isin(NO_REPAIR), POST_TOL, pd.Timedelta(0))
    e["end_eff"] = (e["end_utc"] + pd.to_timedelta(tol)).fillna(w_end).clip(upper=w_end)
    return e.reset_index(drop=True)


def active(times: pd.DatetimeIndex, ev: pd.DataFrame, start="onset_utc", end="end_eff") -> np.ndarray:
    """Boolean matrix (len(times) x len(ev)): event j active at time i, between its ``start`` and ``end`` columns."""
    if not len(ev):
        return np.zeros((len(times), 0), dtype=bool)
    t = times.values[:, None]
    return (t >= ev[start].values[None, :]) & (t <= ev[end].values[None, :])


def in_any_event(times: pd.DatetimeIndex, ev: pd.DataFrame) -> np.ndarray:
    """True while any labeled event (observable or not) is active: such minutes are never treated as healthy."""
    m = active(times, ev)
    return m.any(axis=1) if m.shape[1] else np.zeros(len(times), dtype=bool)


def in_benign(times: pd.DatetimeIndex, ds: Dataset, asset_id: str) -> np.ndarray:
    b = ds.benign[ds.benign["asset_id"] == asset_id] if len(ds.benign) else ds.benign
    out = np.zeros(len(times), dtype=bool)
    for s, e in zip(ts(b["start_utc"]) if len(b) else [], ts(b["end_utc"]) if len(b) else []):
        out |= (times >= s) & (times <= e)
    return out


EARLY = pd.Timedelta(hours=12)
EARLY_SHARE = 2 / 3


def training_labels(times: pd.DatetimeIndex, ds: Dataset, asset_id: str, running: np.ndarray) -> pd.DataFrame:
    """Classifier targets per decision time.

    ``state``: POS (one or more observable events between their detectable time and effective end), NORMAL (outside
    every event window) or EXCL (unobservable events, onset-to-detectable stretches, faults while the pump is stopped).
    POS rows list the active fault types and a weight per type: every event carries total weight 1, two thirds of it
    over its first 12 h after detectable; rows shared by k events give each 1/k; cavitation episodes of one incident
    (same root cause) share one event's weight. COMMUNICATION_LOSS is a target with or without running data.
    """
    n = len(times)
    state = np.full(n, "NORMAL", dtype=object)
    types, weights, ids = [""] * n, [""] * n, [""] * n
    ev = asset_events(ds, asset_id)
    if len(ev):
        state[active(times, ev).any(axis=1)] = "EXCL"
        obs = ev[ev["observable"] & ev["detectable_utc"].notna()].reset_index(drop=True)
        pos = active(times, obs, start="detectable_utc")
        pos &= running[:, None] | (obs["fault_type"] == "COMMUNICATION_LOSS").to_numpy()[None, :]
        incident = np.where(obs["fault_type"] == "CAVITATION", obs["root_cause_event_id"].fillna(obs["event_id"]), obs["event_id"])
        share = 1.0 / pd.Series(incident).map(pd.Series(incident).value_counts()).to_numpy(float)
        w = np.zeros(pos.shape)
        for j in range(pos.shape[1]):
            early = pos[:, j] & (times <= obs["detectable_utc"].iloc[j] + EARLY)
            late = pos[:, j] & ~early
            n1, n2 = early.sum(), late.sum()
            f1, f2 = (EARLY_SHARE, 1 - EARLY_SHARE) if n1 and n2 else (1.0, 1.0)
            if n1:
                w[early, j] = share[j] * f1 / n1
            if n2:
                w[late, j] = share[j] * f2 / n2
        k = pos.sum(axis=1)
        for i in np.flatnonzero(k):
            js = np.flatnonzero(pos[i])
            state[i] = "POS"
            types[i] = "|".join(obs["fault_type"].iloc[js])
            weights[i] = "|".join(f"{w[i, j] / k[i]:.6g}" for j in js)
            ids[i] = "|".join(obs["event_id"].iloc[js])
    return pd.DataFrame({"lab_state": state, "lab_types": types, "lab_weights": weights, "lab_events": ids,
                         "lab_benign": in_benign(times, ds, asset_id)}, index=times)
