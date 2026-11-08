"""Physics consistency, causal prefix, missing evidence, and waveform checks."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("physics_features", Path(__file__).parents[1]/"physics_features.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhysicsFeatureTests(unittest.TestCase):
    def setUp(self):
        self.asset = {"pump_rated_flow_m3_h": 520, "pump_rated_head_m": 26,
                      "pump_shutoff_head_m": 31.2, "pump_bep_efficiency": 0.82,
                      "design_flow_m3_h": 500, "design_head_m": 25.5,
                      "static_head_m": 3, "motor_rated_kw": 55,
                      "motor_rated_current_a": 88.3, "motor_rated_rpm": 1800,
                      "supply_voltage_v": 440, "npsh3_at_bep_m": 4.2,
                      "vibration_alarm_mm_s": 5, "control": "VFD_TEMPERATURE",
                      "min_speed_ratio": 0.5, "bearing_bpfo_order": 3.05,
                      "bearing_bpfi_order": 4.95, "impeller_vanes": 6}
        n = 30
        rho = float(MODULE._water_density(20.0))
        head = 26.0*(1.2-0.2*0.8**2)
        self.frame = pd.DataFrame({"motor_speed_rpm__mean": np.full(n, 1800.),
            "running_fraction": np.ones(n), "flow_m3_h__mean": np.full(n, 416.),
            "suction_pressure_kpa__mean": np.full(n, 20.),
            "discharge_pressure_kpa__mean": np.full(n, 20.+rho*MODULE.G*head/1000.),
            "duty_demand_pct": np.full(n, 80.), "supply_water_temp_c": np.full(n, 20.),
            "ambient_temp_c": np.full(n, 30.), "fluid_temp_c__mean": np.full(n, 20.1),
            "bearing_temp_c__mean": np.linspace(50., 60., n),
            "motor_current_a__mean": np.full(n, 60.), "motor_power_kw__mean": np.full(n, 30.)},
            index=pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"))

    def test_curve_consistency_and_missing_evidence(self):
        feature = MODULE.add_physics_features(self.frame, self.asset)
        np.testing.assert_allclose(feature.phys__head_curve_residual, 0., atol=1e-6)
        np.testing.assert_allclose(feature.phys__head_curve_factor, 1., atol=1e-6)
        missing = MODULE.add_physics_features(self.frame.drop(columns="flow_m3_h__mean"), self.asset)
        self.assertTrue(missing.phys__head_curve_residual.isna().all())
        self.assertTrue(missing.phys__hydraulic_efficiency.isna().all())
        np.testing.assert_allclose(missing.phys__curve_inferred_flow_pu, 416/500, atol=1e-6)
        self.assertTrue(feature.phys__vibration_alarm_fraction.isna().all())

    def test_causal_prefix_and_no_input_mutation(self):
        original = self.frame.copy(deep=True)
        full = MODULE.add_physics_features(self.frame, self.asset)
        for stop in (1, 7, 16, 29):
            prefix = MODULE.add_physics_features(self.frame.iloc[:stop], self.asset)
            pd.testing.assert_frame_equal(prefix, full.iloc[:stop])
        pd.testing.assert_frame_equal(original, self.frame)
        self.assertTrue(full.index.equals(self.frame.index))
        self.assertFalse(np.isinf(full.to_numpy()).any())

    def test_waveform_recovers_order_velocity(self):
        fs = 10240
        t = np.arange(fs)/fs
        fr = 30.
        velocities = {1: 2., 2: 1., 3: 0.5}
        x = sum(v*1e-3*np.sqrt(2)*2*np.pi*(order*fr)/MODULE.G*np.sin(2*np.pi*order*fr*t) for order, v in velocities.items())
        feature = MODULE.extract_snapshot_features(x, fs, fr*60, self.asset)
        for order, expected in velocities.items():
            # One-second Hann leakage puts a small amount of line energy into
            # adjacent 1 Hz bins; integrating 1/f there has a <0.2% bias.
            np.testing.assert_allclose(feature[f"snapshot__velocity_{order}x"], expected, rtol=0.002)
        self.assertAlmostEqual(feature["snapshot__harmonic_2x_1x"], 0.5, places=3)
        self.assertAlmostEqual(feature["snapshot__harmonic_3x_1x"], 0.25, places=3)

    def test_impulse_envelope_selects_bpfo_over_bpfi(self):
        fs = 10240
        fr = 30.
        bpfo = fr*self.asset["bearing_bpfo_order"]*0.995
        pulses = np.zeros(fs)
        pulses[(np.arange(0., 1., 1./bpfo)*fs).astype(int)] = 1.
        kernel_t = np.arange(80)/fs
        x = np.convolve(pulses, np.exp(-kernel_t/0.0008)*np.sin(2*np.pi*2800.*kernel_t))[:fs]
        x += 0.01*np.random.default_rng(832).normal(size=fs)
        feature = MODULE.extract_snapshot_features(x, fs, fr*60, self.asset)
        self.assertGreater(feature["snapshot__envelope_bpfo_fraction"], 3*feature["snapshot__envelope_bpfi_fraction"])
        self.assertGreater(feature["snapshot__envelope_bpfo_prominence"], 10.)

    def test_invalid_waveform_has_missing_features(self):
        result = MODULE.extract_snapshot_features([1., 2.], 10240, 1800, self.asset)
        self.assertTrue(all(np.isnan(v) for v in result.values()))


if __name__ == "__main__":
    unittest.main()
