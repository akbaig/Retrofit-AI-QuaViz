"""The published reference baselines (``retrofit_sim.baselines``), wrapped for the harness with the same train-only
tuning as ``baselines/run_baselines.py``. They are comparison methods only; nothing in our detector uses them."""
from __future__ import annotations

import pandas as pd
from retrofit_sim import baselines as B

from ..data import Dataset
from ..scorer import score_assets


class _ZBaseline:
    """Shared per-asset cache of the baselines' commissioning z-scores (they depend only on the asset)."""
    _z: dict = {}

    def _zf(self, ds: Dataset, ids) -> pd.DataFrame:
        parts = []
        for aid in ids:
            key = (ds.name, aid)
            if key not in self._z:
                sub = ds.subset([aid])
                self._z[key] = B.zscores(B.regime_frame(sub.features, sub.assets))
            parts.append(self._z[key])
        return pd.concat(parts, ignore_index=True)

    def _best(self, grid, run, ds, ids):
        return max(grid, key=lambda p: score_assets(run(ds, ids, p), ds, ids)["composite_score"])


class B0:
    """SCADA alarms as they are; nothing to tune."""
    name = "b0"

    def fit(self, ds, asset_ids):
        return self

    def predict(self, ds, asset_ids):
        return B.b0_scada(ds.alarms, set(asset_ids))

    def describe(self):
        return {}


class B1(_ZBaseline):
    """Mode-aware z: threshold k and persistence tuned on the fit pumps."""
    name = "b1"
    _p: dict = {}
    GRID = [(k, m) for k in [4, 6, 8, 12] for m in [5, 15, 30]]

    def __init__(self):
        self.params = (6.0, 10)

    def _run(self, ds, ids, km):
        out = []
        for aid in ids:
            key = (ds.name, aid, *km)
            if key not in self._p:
                self._p[key] = B.b1_threshold(self._zf(ds, [aid]), *km)
            out.append(self._p[key])
        return pd.concat(out, ignore_index=True)

    def fit(self, ds, asset_ids):
        self.params = self._best(self.GRID, self._run, ds, asset_ids)
        return self

    def predict(self, ds, asset_ids):
        return self._run(ds, asset_ids, self.params)

    def describe(self):
        return {"k": self.params[0], "persist_min": self.params[1]}


class B2(_ZBaseline):
    """CUSUM on the z-scores: decision threshold h tuned on the fit pumps (kappa fixed at 2)."""
    name = "b2"
    GRID = [30, 60, 120, 240]

    def __init__(self):
        self.h = 60

    def _run(self, ds, ids, h):
        return B.b2_cusum(self._zf(ds, ids), 2.0, h)

    def fit(self, ds, asset_ids):
        self.h = self._best(self.GRID, self._run, ds, asset_ids)
        return self

    def predict(self, ds, asset_ids):
        return self._run(ds, asset_ids, self.h)

    def describe(self):
        return {"kappa": 2.0, "h": self.h}


class B3(_ZBaseline):
    """Isolation Forest on fit-pump minutes outside labeled event windows; score quantile tuned on the fit pumps."""
    name = "b3"
    GRID = [0.99, 0.995, 0.999]

    def __init__(self):
        self.q, self._train = 0.995, None

    def fit(self, ds, asset_ids):
        self._train = (self._zf(ds, asset_ids), ds.events[ds.events["asset_id"].isin(set(asset_ids))])
        self.q = self._best(self.GRID, lambda d, ids, q: B.b3_iforest(*self._train, self._zf(d, ids), q=q)[0], ds, asset_ids)
        return self

    def predict(self, ds, asset_ids):
        return B.b3_iforest(*self._train, self._zf(ds, asset_ids), q=self.q)[0]

    def describe(self):
        return {"train_quantile": self.q, "persist_min": 10}
