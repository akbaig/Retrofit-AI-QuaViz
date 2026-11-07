"""Detector methods. Each method has ``name``, ``fit(ds, asset_ids)``, ``predict(ds, asset_ids) -> DataFrame`` in the
submission format (asset_id, alert_time_utc, event_class, fault_type, confidence, evidence, recommended_action) and
``describe() -> dict``."""
from functools import partial

from ..stage3 import CONFIGS
from .baselines import B0, B1, B2, B3
from .quaviz import QuaViz

METHODS = {"b0": B0, "b1": B1, "b2": B2, "b3": B3} | {name: partial(QuaViz, name) for name in CONFIGS}
