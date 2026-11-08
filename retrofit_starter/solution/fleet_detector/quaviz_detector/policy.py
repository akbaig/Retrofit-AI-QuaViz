"""Alert policy: from per-decision class probabilities to submitted alerts.

At each decision only past probabilities are used. ``p_fault = 1 - p_NORMAL`` and the type probabilities are averaged
over the last ``m`` decisions. Decisions while the pump is stopped with data present are skipped (no fault evidence
either way); decisions without data count, so communication loss stays visible.

  * Idle: alert when the averaged p_fault reaches ``on``. The alert names the most probable fault type (and its class)
    and opens an incident.
  * Incident: ends after ``quiet`` consecutive decisions with averaged p_fault below ``off * on``; the next alert
    needs a new crossing.
  * Within an incident (with ``switch`` set), a fault type whose averaged probability rises to ``switch`` gets its own
    alert, at least ``gap`` decisions after the previous alert, but only if it is armed: its probability has stayed
    below ``off * switch`` for ``quiet`` decisions since its last alert (tracked at every decision). A newly appearing
    fault or a new episode of a recurring one (cavitation) gets an alert; two similar types that are both elevated
    and trade the lead do not.
No reminders are sent; the scorer gives them no credit.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .classifier import EVENT_CLASS, FAULTS

PCOLS = [f"p_{f}" for f in FAULTS]


def smooth(probs: pd.DataFrame, m: int) -> dict:
    """Per asset: decision times, averaged p_fault and averaged type probabilities over the last ``m`` decisions."""
    out = {}
    for aid, g in probs.groupby("asset_id", sort=False):
        g = g[g["running"].to_numpy(bool) | (g["rows"].to_numpy() == 0)]
        if len(g):
            out[aid] = (g["decision_time"].to_numpy(), (1.0 - g["p_NORMAL"]).rolling(m, min_periods=1).mean().to_numpy(),
                        g[PCOLS].rolling(m, min_periods=1).mean().to_numpy())
    return out


@dataclass(frozen=True)
class Policy:
    m: int = 3
    on: float = 0.6
    off: float = 0.5
    quiet: int = 6
    switch: float | None = None
    gap: int = 6

    def describe(self) -> dict:
        return asdict(self)

    def alerts(self, probs: pd.DataFrame, smoothed: dict | None = None) -> pd.DataFrame:
        out = []
        for aid, (times, pf, pt) in (smoothed or smooth(probs, self.m)).items():
            lead = pt.argmax(axis=1)
            inc, below, last = False, 0, -10**9
            armed = np.zeros(pt.shape[1], bool)
            low = np.zeros(pt.shape[1], int)
            sw = self.switch if self.switch is not None else 1.0
            for i in range(len(times)):
                low = np.where(pt[i] < self.off * sw, low + 1, 0)
                armed |= low >= self.quiet
                if not inc:
                    if pf[i] >= self.on:
                        k = lead[i]
                        inc, below, last = True, 0, i
                        armed[k], low[k] = False, 0
                        out.append((aid, times[i], k, pt[i, k], pf[i]))
                    continue
                below = below + 1 if pf[i] < self.off * self.on else 0
                if below >= self.quiet:
                    inc = False
                    continue
                if self.switch is None:
                    continue
                if i - last >= self.gap:
                    cand = np.flatnonzero(armed & (pt[i] >= self.switch))
                    if len(cand):
                        k = cand[pt[i, cand].argmax()]
                        armed[k], low[k], last = False, 0, i
                        out.append((aid, times[i], k, pt[i, k], pf[i]))
        df = pd.DataFrame(out, columns=["asset_id", "alert_time_utc", "k", "confidence", "p_fault"])
        df["fault_type"] = [FAULTS[k] for k in df["k"]]
        df["event_class"] = df["fault_type"].map(EVENT_CLASS)
        df["alert_time_utc"] = pd.to_datetime(df["alert_time_utc"], utc=True)
        return df.drop(columns="k")


GRID = dict(m=[3, 6, 12], on=[0.6, 0.7, 0.8, 0.9], off=[0.5], quiet=[6, 12, 36], switch=[None, 0.4, 0.5, 0.6], gap=[6, 36])


def grid(spec: dict = GRID) -> list[Policy]:
    keys = list(spec)
    return [Policy(**dict(zip(keys, v))) for v in itertools.product(*(spec[k] for k in keys))]
