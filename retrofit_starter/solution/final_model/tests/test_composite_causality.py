"""Composition routing and fitted sensor-prefix invariance, no chosen run path."""
import numpy as np
import pandas as pd

from composite import ViewEnsemble
from learning import FAULTS
from sensor_expert import SENSOR_TYPES, SensorExpert
from test_sensor_expert import toy_sensor


class TinyView:
    def __init__(self, offset):
        self.offset = offset

    def transform(self, x, meta):
        return x.copy()

    def probabilities_from_transformed(self, x):
        value = x["level__discharge_pressure_kpa"].fillna(0)
        return pd.DataFrame({fault: np.clip(self.offset+.01*value, 0, 1) for fault in FAULTS}, index=x.index)


def test_ensemble_mean_and_selected_sensor_override():
    x, meta, events = toy_sensor()
    sensor = SensorExpert(max_iter=20).fit(x, meta, events)
    views = [TinyView(.1), TinyView(.3)]
    ensemble = ViewEnsemble(views, sensor, ("SENSOR_NOISE_SPIKES",))
    actual = ensemble.predict_proba(x, meta)
    expected = (views[0].probabilities_from_transformed(x)+views[1].probabilities_from_transformed(x))/2
    shared = sensor.predict(x, meta)
    expected["SENSOR_NOISE_SPIKES"] = shared["SENSOR_NOISE_SPIKES"]
    pd.testing.assert_frame_equal(actual, expected)
    # A sensor model's other outputs must not silently replace unselected heads.
    np.testing.assert_allclose(actual.SENSOR_DRIFT, .2+.01*x.level__discharge_pressure_kpa)


def test_fitted_combined_sensor_expert_prefix_and_future_mutation():
    x, meta, events = toy_sensor()
    sensor = SensorExpert(max_iter=20).fit(x, meta, events)
    ensemble = ViewEnsemble([TinyView(.1), TinyView(.3)], sensor, SENSOR_TYPES)
    stop = 280
    prefix = ensemble.predict_proba(x.iloc[:stop], meta.iloc[:stop])
    full = ensemble.predict_proba(x, meta)
    pd.testing.assert_frame_equal(prefix, full.iloc[:stop])
    mutated = x.copy()
    mutated.iloc[stop:, :] = 10000
    mutated.iloc[stop::2, ::3] = np.nan
    changed = ensemble.predict_proba(mutated, meta)
    pd.testing.assert_frame_equal(prefix, changed.iloc[:stop])
