"""Our detector for the harness: stage-1 deviations, stage-2 window features, stage-3 classifier and alert policy.

``fit(ds, asset_ids)`` takes the official pumps it may learn from; the simulated training seeds are added according
to the configuration. Features come from the stage-2 cache, so ``ds`` is only used by the harness for scoring.
"""
from __future__ import annotations

import pandas as pd

from .. import stage2, stage3
from ..classifier import EVENT_CLASS

ACTIONS = {
    "BEARING_WEAR": "Check bearing temperature and vibration trend; plan bearing inspection and lubrication at the next stop.",
    "SUCTION_STRAINER_CLOGGING": "Check suction pressure against the strainer differential; plan strainer cleaning.",
    "CAVITATION": "Check suction head and tank level; reduce flow or raise suction pressure if possible.",
    "DISCHARGE_RESTRICTION": "Check discharge valve position and downstream line for a blockage.",
    "MISALIGNMENT": "Plan a shaft alignment check at the next stop.",
    "IMBALANCE": "Plan an impeller and rotor balance inspection.",
    "IMPELLER_WEAR": "Compare head and flow with the pump curve; plan an impeller inspection.",
    "AIR_ENTRAINMENT": "Check suction tank level, vortexing and suction-side leaks.",
    "SENSOR_DRIFT": "Verify the drifting sensor against a reference instrument; recalibrate.",
    "SENSOR_STUCK": "Check the frozen sensor and its wiring; do not act on its reading.",
    "SENSOR_CALIBRATION_SHIFT": "Verify the sensor calibration against a reference instrument.",
    "SENSOR_NOISE_SPIKES": "Check sensor wiring, grounding and shielding.",
    "COMMUNICATION_LOSS": "Check the data link and gateway; the pump state is unknown until data returns.",
}


def _evidence(row: pd.Series, p_fault: float) -> str:
    z = row.filter(regex=r"^z[AB]_.*_m30$").astype(float).dropna()
    top = z.reindex(z.abs().sort_values(ascending=False).index[:3])
    devs = ", ".join(f"{k[:-4]} {v:+.1f}" for k, v in top.items())
    return f"p_fault {p_fault:.2f}; largest 30-min deviations (normal-model sigmas): {devs or 'none (no data)'}"


class QuaViz:
    def __init__(self, config: str = "quaviz"):
        self.cfg = stage3.CONFIGS[config]
        self.name = config

    def fit(self, ds, asset_ids):
        official = sorted(a for a in asset_ids if a.startswith("PUMP"))
        self.ids = stage3.training_ids(self.cfg, official)
        self.clf = stage3.fit_classifier(self.cfg, self.ids)
        self.policy, _ = stage3.tune(self.cfg, official)
        return self

    def predict(self, ds, asset_ids):
        return self.predict_table(stage2.load(asset_ids))

    def predict_table(self, tab: pd.DataFrame) -> pd.DataFrame:
        """Alerts from stage-2 tables built outside the cache (``stage2.pump_table``)."""
        al = self.policy.alerts(self.clf.predict_proba(tab))
        if not len(al):
            return pd.DataFrame(columns=["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence",
                                         "recommended_action"])
        look = tab.set_index(["asset_id", "decision_time"])
        al["evidence"] = [_evidence(look.loc[(a, t)], pf) for a, t, pf in zip(al["asset_id"], al["alert_time_utc"], al["p_fault"])]
        al["recommended_action"] = al["fault_type"].map(ACTIONS)
        al["event_class"] = al["fault_type"].map(EVENT_CLASS)
        al["confidence"] = al["confidence"].round(3)
        al["alert_time_utc"] = al["alert_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return al[["asset_id", "alert_time_utc", "event_class", "fault_type", "confidence", "evidence", "recommended_action"]]

    def describe(self):
        return {"config": self.cfg.name, "training_pumps": len(self.ids), "features": len(self.clf.cols),
                "policy": self.policy.describe()}
