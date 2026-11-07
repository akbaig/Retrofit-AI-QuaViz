"""Stage 3: fault classifier over the stage-2 tables.

One gradient-boosted multiclass model over NORMAL and the 13 fault types; the event class follows from the type.

Training rows and weights (``design``):
  * POS rows (every ``POS_EVERY``-th decision) are repeated once per active fault type with that type's weight (``labels.training_labels``), so every
    event carries the same total weight whatever its length;
  * NORMAL rows (every ``NORMAL_EVERY``-th decision outside all event windows) share weight 1 per pump, with rows in
    benign windows (starts, stops, load changes, storms, voltage sags) counted ``BENIGN_WEIGHT`` times;
  * NORMAL as a whole gets ``NORMAL_SHARE`` times the total fault weight;
  * official pumps count ``OFFICIAL_WEIGHT`` times a simulated pump.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .stage2 import DEV_PREFIXES, LABELS, META

FAULTS = ["AIR_ENTRAINMENT", "BEARING_WEAR", "CAVITATION", "COMMUNICATION_LOSS", "DISCHARGE_RESTRICTION", "IMBALANCE",
          "IMPELLER_WEAR", "MISALIGNMENT", "SENSOR_CALIBRATION_SHIFT", "SENSOR_DRIFT", "SENSOR_NOISE_SPIKES",
          "SENSOR_STUCK", "SUCTION_STRAINER_CLOGGING"]
CLASSES = ["NORMAL"] + FAULTS
EVENT_CLASS = {f: "SENSOR" if f.startswith("SENSOR") or f == "COMMUNICATION_LOSS" else "MACHINE" for f in FAULTS}
OFFICIAL_WEIGHT = 5.0
BENIGN_WEIGHT = 3.0
NORMAL_SHARE = 1.0
NORMAL_EVERY = 3
POS_EVERY = 2


def feature_columns(table: pd.DataFrame, deviations: bool = True) -> list[str]:
    skip = set(META) | set(LABELS) | {"decision_time"}
    cols = [c for c in table.columns if c not in skip and table[c].dtype != object]
    return cols if deviations else [c for c in cols if not c.startswith(DEV_PREFIXES)]


def design(table: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    official = table["asset_id"].str.startswith("PUMP").to_numpy()
    src = np.where(official, OFFICIAL_WEIGHT, 1.0)
    state = table["lab_state"].to_numpy()
    X = table[cols].to_numpy(np.float32)

    pos_in_pump = table.groupby("asset_id").cumcount().to_numpy()
    pi = np.flatnonzero((state == "POS") & (pos_in_pump % POS_EVERY == 0))
    types = table["lab_types"].to_numpy()[pi]
    wts = table["lab_weights"].to_numpy()[pi]
    rep, ys, ws = [], [], []
    for i, t, w in zip(pi, types, wts):
        for tt, ww in zip(t.split("|"), w.split("|")):
            rep.append(i)
            ys.append(CLASSES.index(tt))
            ws.append(float(ww) * src[i])
    rep, ys, ws = np.array(rep, int), np.array(ys, int), np.array(ws)

    ni = np.flatnonzero((state == "NORMAL") & (pos_in_pump % NORMAL_EVERY == 0))
    nw = np.where(table["lab_benign"].to_numpy(bool)[ni], BENIGN_WEIGHT, 1.0)
    per_pump = pd.Series(nw).groupby(table["asset_id"].to_numpy()[ni]).transform("sum").to_numpy()
    nw = nw / per_pump * src[ni]
    nw *= NORMAL_SHARE * ws.sum() / nw.sum()

    idx = np.concatenate([rep, ni])
    y = np.concatenate([ys, np.zeros(len(ni), int)])
    w = np.concatenate([ws, nw])
    return X[idx], y, w / w.mean()


class FaultClassifier:
    def __init__(self, deviations: bool = True, max_iter: int = 150, seed: int = 0):
        self.deviations, self.max_iter, self.seed = deviations, max_iter, seed

    def fit(self, table: pd.DataFrame) -> "FaultClassifier":
        self.cols = feature_columns(table, self.deviations)
        X, y, w = design(table, self.cols)
        self.m = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, max_bins=63, max_leaf_nodes=31,
                                                min_samples_leaf=40, l2_regularization=1.0, max_features=0.5,
                                                early_stopping=False, random_state=self.seed).fit(X, y, sample_weight=w)
        self.n_rows = len(y)
        return self

    def predict_proba(self, table: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities per decision, one column per class (``p_NORMAL``, ``p_<FAULT>``)."""
        P = np.zeros((len(table), len(CLASSES)))
        P[:, self.m.classes_] = self.m.predict_proba(table.reindex(columns=self.cols).to_numpy(np.float32))
        out = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASSES], index=table.index)
        out.insert(0, "asset_id", table["asset_id"].values)
        out.insert(1, "decision_time", table["decision_time"].values)
        out.insert(2, "running", table["running"].values.astype(bool))
        out.insert(3, "rows", table["rows"].values)
        return out
