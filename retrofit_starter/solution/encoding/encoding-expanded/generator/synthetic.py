"""Fresh, bounded simulator augmentation through the published observation path.

Only PUMP-01..07 engineering specifications and profile defaults are reused. Every
published scenario is discarded. Model inputs are public telemetry aggregates,
public context and public asset metadata; no latent/truth table is exported.

CLI examples (run with the experiment's Python environment)::
    python synthetic.py --role train
    python synthetic.py --role development
    python synthetic.py --role train_balanced --days 2
    python synthetic.py --role train_crossed --days 2
    python synthetic.py --role train_sensor_crossed --days 1.5
    python synthetic.py --role holdout --freeze-manifest results/frozen_manifest.json

The holdout is reserved now and generated only after the caller has frozen its
pipeline. This module never reads its generated holdout events back from disk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from paths import HERE, SOURCE as DEFAULT_SOURCE
DEFAULT_OUTPUT = HERE / "synthetic_data"
TRAIN_ASSETS = tuple(f"PUMP-{i:02d}" for i in range(1, 8))
ROLE_SEEDS = {"train": (813107, 813109, 813121), "development": (913127,),
              "holdout": (1013137, 1013141, 1013153)}
BALANCED_SEEDS = (1213139, 1213151)
CROSSED_SEEDS = (1313161,)
SENSOR_CROSSED_SEEDS = (1413181,)
SENSOR_CROSSED_STRATA = (
    ("PUMP-02", "FIXED_SPEED", "T2_FULL", "2024-05-15T00:00:00"),
    ("PUMP-07", "VFD_TEMPERATURE", "T2_FULL", "2024-08-15T00:00:00"),
    ("PUMP-04", "VFD_PRESSURE", "T1_RETROFIT", "2024-06-17T00:00:00"),
)
SENSOR_CROSSED_TARGETS = (
    ("SENSOR_DRIFT", "motor_current_a"),
    ("SENSOR_DRIFT", "motor_power_kw"),
    ("SENSOR_NOISE_SPIKES", "motor_current_a"),
    ("SENSOR_NOISE_SPIKES", "motor_power_kw"),
    ("SENSOR_NOISE_SPIKES", "vibration_accel_rms_g"),
    ("SENSOR_NOISE_SPIKES", "vibration_crest_factor"),
    ("HEALTHY", None),
)
CROSSED_STRATA = (
    ("PUMP-02", "FIXED_SPEED", "T2_FULL", False),
    ("PUMP-07", "VFD_TEMPERATURE", "T2_FULL", False),
    ("PUMP-04", "FIXED_SPEED", "T2_FULL", False),
    ("PUMP-05", "VFD_PRESSURE", "T1_RETROFIT", True),
)
BALANCED_TRACKS = (
    ("PUMP-02", "T2_FULL", "BEARING_WEAR"),
    ("PUMP-01", "T1_RETROFIT", "BEARING_WEAR"),
    ("PUMP-04", "T2_FULL", "BEARING_WEAR"),
    ("PUMP-02", "T2_FULL", "CAVITATION"),
    ("PUMP-07", "T1_RETROFIT", "CAVITATION"),
    ("PUMP-06", "T2_FULL", "CAVITATION"),
    ("PUMP-01", "T1_RETROFIT", "AIR_ENTRAINMENT"),
    ("PUMP-07", "T2_FULL", "AIR_ENTRAINMENT"),
    ("PUMP-02", "T2_FULL", "MISALIGNMENT"),
    ("PUMP-06", "T1_RETROFIT", "MISALIGNMENT"),
    ("PUMP-04", "T2_FULL", "IMBALANCE"),
    ("PUMP-01", "T1_RETROFIT", "IMBALANCE"),
    ("PUMP-03", "T0_LEGACY", "IMPELLER_WEAR"),
    ("PUMP-05", "T0_LEGACY", "DISCHARGE_RESTRICTION"),
    ("PUMP-04", "T1_RETROFIT", "HEALTHY"),
)
MACHINES = ("BEARING_WEAR", "SUCTION_STRAINER_CLOGGING", "DISCHARGE_RESTRICTION",
            "MISALIGNMENT", "IMBALANCE", "IMPELLER_WEAR", "AIR_ENTRAINMENT")
CROSSED_CLASSES = (*MACHINES, "CAVITATION", "HEALTHY")
SENSORS = ("SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT",
           "SENSOR_NOISE_SPIKES", "COMMUNICATION_LOSS")
EVENT_TIME_COLS = ("onset_ms", "detectable_ms", "action_deadline_ms", "end_ms")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_source(source_root: Path):
    src = str(source_root.resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def training_templates(source_root: Path = DEFAULT_SOURCE) -> list[dict]:
    """Load training engineering specs without using any published fault plan."""
    _install_source(source_root)
    from retrofit_sim.dataset import enrich_asset
    from retrofit_sim.fleet import _deep_merge, default_limits
    cfg = yaml.safe_load((source_root / "configs/fleet_public.yaml").read_text())
    templates = []
    for raw in cfg["assets"]:
        if raw["asset_id"] not in TRAIN_ASSETS:
            continue
        assert raw["split"] == "train"
        spec = {key: value for key, value in raw.items() if key != "scenario"}
        spec = _deep_merge(cfg["profiles"][spec["profile"]], spec)
        spec.pop("scenario", None)
        spec["limits"] = default_limits(spec)
        spec["_limit_overrides"] = []
        templates.append(enrich_asset(spec))
    assert {a["asset_id"] for a in templates} == set(TRAIN_ASSETS)
    return sorted(templates, key=lambda a: a["asset_id"])


def _machine_params(rng, typ: str, duration_h: float) -> dict:
    if typ == "BEARING_WEAR":
        return {"rate_per_h": float(rng.uniform(.008, .06)),
                "t_fail_h": float(duration_h * rng.uniform(.60, 1.50)), "w_max": float(rng.uniform(.65, .95))}
    if typ == "SUCTION_STRAINER_CLOGGING":
        return {"shape": "linear", "rate_per_h": float(rng.uniform(.06, .36)), "c_max": float(rng.uniform(2.0, 7.0))}
    if typ == "DISCHARGE_RESTRICTION":
        return {"shape": "exp", "rate_per_h": float(rng.choice([.025, .08, .25, 1.5])), "extra_max": float(rng.uniform(.25, 1.8))}
    if typ == "MISALIGNMENT":
        return {"level": float(rng.uniform(.20, .75)), "growth_per_h": float(rng.uniform(0, .006))}
    if typ == "IMBALANCE":
        return {"shape": "linear", "rate_per_h": float(rng.uniform(.015, .085)),
                "u0": float(rng.uniform(0, .15)), "u_max": float(rng.uniform(.50, 1.7))}
    if typ == "IMPELLER_WEAR":
        return {"rate_per_h": float(rng.uniform(.002, .009)), "iw0": 0.0, "iw_max": float(rng.uniform(.04, .15))}
    if typ == "AIR_ENTRAINMENT":
        return {"alpha_max": float(rng.uniform(.008, .05))}
    raise ValueError(typ)


def _sensor_params(rng, typ, channel) -> dict:
    # Engineering-unit scales come from the actual installed channel, making
    # drift/spike severity comparable across current, pressure and temperature.
    scale = max(channel.sigma * 4.0, channel.resolution * 4.0)
    sign = float(rng.choice([-1., 1.]))
    if typ == "SENSOR_DRIFT":
        return {"drift_per_h": sign * scale * float(rng.uniform(.025, .18))}
    if typ == "SENSOR_CALIBRATION_SHIFT":
        return {"step": sign * scale * float(rng.uniform(.8, 3.5))}
    if typ == "SENSOR_NOISE_SPIKES":
        return {"noise_mult": float(rng.uniform(2.5, 7)), "spike_rate": float(rng.uniform(.001, .02)),
                "spike_size": scale * float(rng.uniform(2, 8))}
    return {}


def randomized_asset(template, seed: int, index: int, role: str, days: float) -> dict:
    """Generate independent healthy, single and compound trajectories.

    The role/index determines a coverage stratum, not an inference feature.
    All onset, severity, channel, environment and noise draws are independent of
    published event plans. Cavitation is emergent physics, never a direct label.
    """
    from retrofit_sim.fleet import default_limits
    from retrofit_sim.sensors import channel_specs
    rng = np.random.default_rng([seed, index, 271828])
    a = copy.deepcopy(template)
    a["template_asset_id"] = a["asset_id"]
    a["asset_id"] = f"SYN-{role.upper()}-{seed}-{index + 1:02d}"
    a["split"] = role
    a["site"] = f"Fresh synthetic {role} fleet"
    a["start_utc"] = f"2024-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 20)):02d}T00:00:00"
    a["sample_drop_prob"] = float(rng.uniform(.0005, .004))
    a["demand_noise"] = float(rng.uniform(.02, .07))
    a["voltage_sags_per_week"] = float(rng.uniform(.4, 2.5))
    a["routine_lockouts_per_week"] = float(rng.uniform(.5, 1.8))
    a.setdefault("context", {})
    if a["profile"].startswith("SHIP"):
        a["context"]["storm_at_h"] = float(rng.uniform(.10, .80) * days * 24)
        a["context"]["hs_mean"] = float(rng.uniform(.6, 1.8))
        a["context"]["standby_prob"] = float(rng.uniform(0, .20))
    # Training/dev stay close to training engineering specs. The final holdout
    # deliberately changes ratings and some sensor tiers as a domain check.
    spread = .15 if role == "holdout" else .04
    flow_scale, head_scale = rng.uniform(1 - spread, 1 + spread, 2)
    for key in ("q_bep_m3h", "q_design_m3h"):
        a[key] *= float(flow_scale)
    for key in ("h_bep_m", "h_design_m", "system_static_m", "dp_setpoint_m"):
        if key in a:
            a[key] *= float(head_scale)
    # Hydraulic power is proportional to flow x head; retain nameplate margin.
    a["motor_kw"] *= float(flow_scale * head_scale)
    a["rating_scale_flow"] = float(flow_scale)
    a["rating_scale_head"] = float(head_scale)
    a["vib_base_mm_s"] *= float(rng.uniform(.90, 1.10))
    a["accel_base_g"] *= float(rng.uniform(.90, 1.10))
    if role == "holdout" and index % 3 == 1:
        a["instrumentation_tier"] = str(rng.choice(["T0_LEGACY", "T1_RETROFIT", "T2_FULL"]))
    if a["instrumentation_tier"] == "T1_RETROFIT":
        a["wireless_update_s"] = int(rng.choice([60, 120, 300]))
    a["limits"] = default_limits(a)
    a["scenario"] = []
    # One fully healthy trajectory per seed, rotated independently by seed.
    healthy_index = seed % len(TRAIN_ASSETS)
    if index == healthy_index:
        a["scenario_kind"] = "healthy"
        return a

    hours = days * 24
    machine_count = 1 if index % 3 == 0 else 2 if index % 3 == 1 else 3
    first = (index + seed % len(MACHINES)) % len(MACHINES)
    picks = [MACHINES[first]]
    picks += list(rng.choice([m for m in MACHINES if m not in picks], machine_count - 1, replace=False))
    # Preserve an initial healthy history, vary fault starts and overlaps.
    for j, typ in enumerate(picks):
        onset = float(rng.uniform(.12, .52) * hours)
        duration = float(rng.uniform(.18, .48) * hours)
        end = min(hours - 1.5, onset + duration) if rng.random() < .8 else None
        e = {"type": str(typ), "onset_h": onset, "jitter_h": 0,
             "params": _machine_params(rng, str(typ), duration)}
        if end is not None:
            e.update(end_h=end, repair_h=float(rng.uniform(.3, 2.0)))
        a["scenario"].append(e)

    specs = channel_specs(a)
    eligible = [c for c in specs if c != "ambient_temp_c"]
    sensor_count = 1 + int(index % 2 == 0)
    used_tags = set()
    for j in range(sensor_count):
        typ = SENSORS[(index + j + seed % len(SENSORS)) % len(SENSORS)]
        onset = float(rng.uniform(.10, .74) * hours)
        duration = float(rng.uniform(2, min(18, hours * .25)))
        tag = "GATEWAY" if typ == "COMMUNICATION_LOSS" and rng.random() < .5 else str(rng.choice([c for c in eligible if c not in used_tags]))
        used_tags.add(tag)
        e = {"type": typ, "onset_h": onset, "end_h": min(hours - .5, onset + duration),
             "jitter_h": 0, "sensor_tag": tag,
             "params": {} if tag == "GATEWAY" else _sensor_params(rng, typ, specs[tag])}
        a["scenario"].append(e)
    # A context-side suction disturbance provides cavitation examples without
    # making a fictitious latent CAVITATION knob or changing observed channels.
    if index % 3 == 1 and a["profile"] != "SHIP_LT_FRESHWATER":
        onset = float(rng.uniform(.35, .65) * hours)
        a["scenario"].append({"type": "SUCTION_LEVEL_DROP", "onset_h": onset,
                              "end_h": min(hours - 1, onset + rng.uniform(5, 12)), "jitter_h": 0,
                              "params": {"drop_m": float(rng.uniform(3, 6)), "ramp_h": float(rng.uniform(.2, 1.5))}})
    a["scenario_kind"] = "single_machine" if machine_count == 1 else "compound"
    return a


def reserve_seeds(output_root: Path = DEFAULT_OUTPUT) -> dict:
    """Persist the complete role split before any evaluation samples exist."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "seed_manifest.json"
    manifest = {"version": 1, "roles": {k: list(v) for k, v in ROLE_SEEDS.items()},
                "templates": list(TRAIN_ASSETS), "days": 3,
                "holdout_status": "reserved; generation requires a frozen pipeline manifest",
                "holdout_domain_variation": "flow/head +/-15%; power scales with flow*head; seasons/context/tier variation",
                "source_reuse": "profile defaults and training engineering specs; all published scenarios discarded"}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError("Existing reserved seed manifest differs; refusing to rewrite evaluation split")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def balanced_asset(templates: list[dict], seed: int, index: int, days: float) -> dict:
    """Coverage-targeted training cases with observed-only inference inputs.

    CAVITATION strata apply real suction-loss/level changes and receive labels
    only if the published physics/labeling code produces an episode. Sensor
    disturbances occur on installed channels at independent timings.
    """
    from retrofit_sim.fleet import default_limits
    from retrofit_sim.sensors import channel_specs
    template_id, tier, target = BALANCED_TRACKS[index]
    template = next(t for t in templates if t["asset_id"] == template_id)
    a = randomized_asset(template, seed, index, "train_balanced", days)
    a["instrumentation_tier"] = tier
    a["scenario"] = []
    a["scenario_kind"] = f"balanced_{target.lower()}"
    a["target_coverage_stratum"] = target
    rng = np.random.default_rng([seed, index, 314159])
    if tier == "T1_RETROFIT":
        a["wireless_update_s"] = int(rng.choice([60, 120, 300]))
    a["limits"] = default_limits(a)
    if target == "HEALTHY":
        a["scenario_kind"] = "healthy"
        return a

    hours = days * 24
    onset = float(rng.uniform(.18, .30) * hours)
    end = float(rng.uniform(.78, .92) * hours)
    duration = end - onset
    if target == "CAVITATION":
        # Pump suction strainer plugging and reduced static inlet head are both
        # explicitly supported causes of cavitation in the published simulator.
        # Parameter ranges deliberately span mild and clear physical symptoms.
        a["scenario"].append({"type": "SUCTION_STRAINER_CLOGGING", "onset_h": onset,
                              "end_h": end, "jitter_h": 0, "repair_h": .7,
                              "params": {"shape": "linear", "rate_per_h": float(rng.uniform(.22, .40)),
                                         "c_max": float(rng.uniform(4.5, 7.5))}})
        drop = float(rng.uniform(4.5, 7.5)) if a["profile"] != "PROCESS_TRANSFER" else float(rng.uniform(2.8, 4.5))
        a["scenario"].append({"type": "SUCTION_LEVEL_DROP", "onset_h": onset + float(rng.uniform(1, 3)),
                              "end_h": end, "jitter_h": 0,
                              "params": {"drop_m": drop, "ramp_h": float(rng.uniform(.5, 2))}})
    else:
        params = _machine_params(rng, target, duration)
        if target == "BEARING_WEAR":
            params.update(rate_per_h=float(rng.uniform(.015, .055)),
                          t_fail_h=duration * float(rng.uniform(.65, 1.05)),
                          w_max=float(rng.uniform(.72, .95)))
        elif target == "AIR_ENTRAINMENT":
            params["alpha_max"] = float(rng.uniform(.025, .065))
        elif target == "MISALIGNMENT":
            params.update(level=float(rng.uniform(.38, .85)), growth_per_h=float(rng.uniform(0, .004)))
        elif target == "IMBALANCE":
            params.update(rate_per_h=float(rng.uniform(.035, .085)), u_max=float(rng.uniform(.8, 1.8)))
        a["scenario"].append({"type": target, "onset_h": onset, "end_h": end,
                              "jitter_h": 0, "repair_h": float(rng.uniform(.5, 2)), "params": params})

    # Two thirds of tracks carry a separately timed sensor fault. The other
    # third supplies uncorrupted machine signatures, avoiding a necessary
    # association between machine failures and instrumentation faults.
    if index % 3 != 0:
        specs = channel_specs(a)
        preferred = ("motor_current_a", "bearing_temp_c", "suction_pressure_kpa")
        have = [tag for tag in preferred if tag in specs]
        sensor_rank = sum(k % 3 != 0 for k in range(index))
        tag = have[sensor_rank % len(have)]
        typ = "SENSOR_DRIFT" if sensor_rank % 2 == 0 else "SENSOR_CALIBRATION_SHIFT"
        s_on = float(rng.uniform(.08, .62) * hours)
        s_end = min(hours - .4, s_on + float(rng.uniform(8, 18)))
        sensor_params = _sensor_params(rng, typ, specs[tag])
        if typ == "SENSOR_DRIFT":
            # Sparse, quantized wireless temperatures need a drift larger than
            # several resolution steps to supply an identifiable training case.
            c = specs[tag]
            sigma10 = max(c.resolution, c.sigma / np.sqrt(max(1, 600 / c.update_s)))
            rate = sensor_params["drift_per_h"]
            sensor_params["drift_per_h"] = float(np.sign(rate) * max(abs(rate), 5 * sigma10 / (s_end - s_on)))
        a["scenario"].append({"type": typ, "sensor_tag": tag, "onset_h": s_on,
                              "end_h": s_end, "jitter_h": 0,
                              "params": sensor_params})
    return a


def reserve_balanced_seeds(output_root: Path, seeds: tuple[int, ...], days: float) -> dict:
    """Add a training-only seed manifest without touching the frozen split."""
    path = output_root / "supplemental_training_seed_manifest.json"
    manifest = {"version": 1, "role": "train_balanced", "purpose": "training augmentation only",
                "seeds": list(seeds), "days": days, "tracks_per_seed": len(BALANCED_TRACKS),
                "coverage_strata": [list(s) for s in BALANCED_TRACKS],
                "selection_basis": "published simulator physics and original synthetic training class coverage",
                "validation_or_test_data_used": False,
                "reserved_evaluation_seed_manifest_sha256": _sha256(output_root / "seed_manifest.json")}
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing supplemental training seed manifest differs")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def crossed_asset(templates: list[dict], seed: int, index: int, days: float) -> dict:
    """Cross every machine class with every engineering/observation stratum.

    Planned target classes have identical profile, control, tier, physical
    template and configured channel-mask coverage. Observable event labels
    remain physics-dependent. CAVITATION is an emergent hydraulic outcome,
    not a direct latent state or feature. A machine fault need not have a
    concurrent sensor fault.
    """
    from retrofit_sim.fleet import default_limits
    stratum_index = index // len(CROSSED_CLASSES)
    template_id, control, tier, remove_speed = CROSSED_STRATA[stratum_index]
    target = CROSSED_CLASSES[index % len(CROSSED_CLASSES)]
    template = next(t for t in templates if t["asset_id"] == template_id)
    # Share engineering and instrumentation parameters across all nine classes
    # in each cohort. Unique IDs still produce independent source RNG streams.
    a = randomized_asset(template, seed, stratum_index, "train_crossed", days)
    a["asset_id"] = f"SYN-TRAIN_CROSSED-{seed}-{index + 1:02d}"
    # Use the same season for every class within a control cohort. In the warm
    # VFD ship cohort, cooling demand reaches higher flows, letting a physical
    # suction disturbance produce cavitation without inventing a fault knob.
    months = (5, 8, 11, 6)
    a["start_utc"] = f"2024-{months[stratum_index]:02d}-15T00:00:00"
    a["control"] = control
    a["instrumentation_tier"] = tier
    a["missing_channels"] = ["motor_speed_rpm"] if remove_speed else []
    a["scenario"] = []
    a["scenario_kind"] = f"crossed_{target.lower()}"
    a["target_coverage_stratum"] = target
    rng = np.random.default_rng([seed, index, 161803])
    if tier == "T1_RETROFIT":
        a["wireless_update_s"] = 120
    a["limits"] = default_limits(a)
    if target == "HEALTHY":
        a["scenario_kind"] = "healthy"
        return a
    hours = days * 24
    onset = float(rng.uniform(.18, .30) * hours)
    end = float(rng.uniform(.82, .94) * hours)
    duration = end - onset
    if target == "CAVITATION":
        a["scenario"].append({"type": "SUCTION_STRAINER_CLOGGING", "onset_h": onset,
                              "end_h": end, "jitter_h": 0, "repair_h": .7,
                              "params": {"shape": "linear", "rate_per_h": float(rng.uniform(.22, .4)),
                                         "c_max": float(rng.uniform(4.5, 7.5))}})
        a["scenario"].append({"type": "SUCTION_LEVEL_DROP", "onset_h": onset + float(rng.uniform(1, 3)),
                              "end_h": end, "jitter_h": 0,
                              "params": {"drop_m": float(rng.uniform(5.5, 8.0)),
                                         "ramp_h": float(rng.uniform(.5, 2))}})
    else:
        params = _machine_params(rng, target, duration)
        if target == "BEARING_WEAR":
            params.update(rate_per_h=float(rng.uniform(.015, .045)),
                          t_fail_h=duration * float(rng.uniform(.72, 1.05)),
                          w_max=float(rng.uniform(.75, .95)))
        elif target == "AIR_ENTRAINMENT":
            params["alpha_max"] = float(rng.uniform(.025, .06))
        elif target == "MISALIGNMENT":
            params.update(level=float(rng.uniform(.4, .8)), growth_per_h=float(rng.uniform(0, .004)))
        elif target == "IMBALANCE":
            params.update(rate_per_h=float(rng.uniform(.035, .085)), u_max=float(rng.uniform(.8, 1.7)))
        elif target == "SUCTION_STRAINER_CLOGGING":
            params.update(rate_per_h=float(rng.uniform(.15, .3)), c_max=float(rng.uniform(2.5, 5)))
        a["scenario"].append({"type": target, "onset_h": onset, "end_h": end, "jitter_h": 0,
                              "repair_h": float(rng.uniform(.5, 2)), "params": params})
    return a


def reserve_crossed_seeds(output_root: Path, seeds: tuple[int, ...], days: float) -> dict:
    path = output_root / "crossed_training_seed_manifest.json"
    manifest = {"version": 1, "role": "train_crossed", "purpose": "training augmentation only",
                "seeds": list(seeds), "days": days,
                "tracks_per_seed": len(CROSSED_STRATA) * len(CROSSED_CLASSES),
                "classes": list(CROSSED_CLASSES), "coverage_strata": [list(s) for s in CROSSED_STRATA],
                "selection_basis": "full factorial machine class by control/profile/instrumentation coverage",
                "validation_or_test_data_used": False,
                "reserved_evaluation_seed_manifest_sha256": _sha256(output_root / "seed_manifest.json")}
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing crossed training seed manifest differs")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sensor_crossed_asset(templates: list[dict], seed: int, index: int, days: float) -> dict:
    """Independent sensor corruption crossed against identical cohort metadata."""
    from retrofit_sim.fleet import default_limits
    from retrofit_sim.sensors import channel_specs
    stratum = index // len(SENSOR_CROSSED_TARGETS)
    target, tag = SENSOR_CROSSED_TARGETS[index % len(SENSOR_CROSSED_TARGETS)]
    template_id, control, tier, start = SENSOR_CROSSED_STRATA[stratum]
    template = next(t for t in templates if t["asset_id"] == template_id)
    a = randomized_asset(template, seed, stratum, "train_sensor_crossed", days)
    a.update(asset_id=f"SYN-TRAIN_SENSOR_CROSSED-{seed}-{index + 1:02d}",
             control=control, instrumentation_tier=tier, start_utc=start,
             missing_channels=[], scenario=[],
             scenario_kind="healthy" if target == "HEALTHY" else f"sensor_crossed_{target.lower()}_{tag}")
    if tier == "T1_RETROFIT":
        a["wireless_update_s"] = 120
    a["limits"] = default_limits(a)
    if target == "HEALTHY":
        return a
    specs = channel_specs(a)
    assert tag in specs
    c = specs[tag]
    rng = np.random.default_rng([seed, index, 173205])
    hours = days * 24
    onset = float(rng.uniform(.18, .29) * hours)
    end = float(rng.uniform(.76, .9) * hours)
    duration = end - onset
    if target == "SENSOR_DRIFT":
        # Slowly accumulate a few percent of transmitter range over 17-25h.
        # The sign and exact engineering scale are independent of class/cohort.
        displacement = max(c.resolution * float(rng.uniform(10, 30)),
                           (c.hi - c.lo) * float(rng.uniform(.015, .06)))
        params = {"drift_per_h": float(rng.choice([-1, 1]) * displacement / duration)}
    else:
        # The published observation model applies extra noise at every second,
        # then performs sparse sampling. Long intervals span many actual T1
        # transmissions instead of relying on a single spike between updates.
        params = {"noise_mult": float(rng.uniform(3, 8)),
                  "spike_rate": float(rng.uniform(.005, .025)),
                  "spike_size": max(c.resolution * 10,
                                    (c.hi - c.lo) * float(rng.uniform(.003, .015)))}
    a["scenario"] = [{"type": target, "sensor_tag": tag, "onset_h": onset,
                       "end_h": end, "jitter_h": 0, "params": params}]
    return a


def reserve_sensor_crossed_seeds(output_root: Path, seeds: tuple[int, ...], days: float) -> dict:
    path = output_root / "sensor_crossed_training_seed_manifest.json"
    manifest = {"version": 1, "role": "train_sensor_crossed", "purpose": "training augmentation only",
                "seeds": list(seeds), "days": days,
                "tracks_per_seed": len(SENSOR_CROSSED_STRATA) * len(SENSOR_CROSSED_TARGETS),
                "targets": [list(s) for s in SENSOR_CROSSED_TARGETS],
                "coverage_strata": [list(s) for s in SENSOR_CROSSED_STRATA],
                "selection_basis": "training label coverage gaps and generic channel engineering scales",
                "validation_or_development_event_configurations_used": False,
                "reserved_evaluation_seed_manifest_sha256": _sha256(output_root / "seed_manifest.json")}
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing sensor crossed training seed manifest differs")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_role(source_root: Path = DEFAULT_SOURCE, output_root: Path = DEFAULT_OUTPUT,
                  role: str = "train", seeds=None, days: float = 3,
                  freeze_manifest: Path | None = None, log=print) -> dict:
    """Write observed feature/context/asset tables plus isolated target labels."""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    role_seeds = dict(ROLE_SEEDS, train_balanced=BALANCED_SEEDS, train_crossed=CROSSED_SEEDS,
                      train_sensor_crossed=SENSOR_CROSSED_SEEDS)
    if role not in role_seeds:
        raise ValueError(f"Unknown role: {role}")
    if days < 1:
        raise ValueError("At least one day is needed for realistic schedules and recovery periods")
    reserve_seeds(output_root)
    seeds = tuple(seeds or role_seeds[role])
    others = {s for k, vv in role_seeds.items() if k != role for s in vv}
    if set(seeds) & others:
        raise ValueError("A seed cannot cross training/development/holdout roles")
    if role == "holdout" and (freeze_manifest is None or not Path(freeze_manifest).is_file()):
        raise ValueError("Holdout generation requires an existing frozen pipeline manifest")
    if role == "train_balanced":
        reserve_balanced_seeds(output_root, seeds, days)
    elif role == "train_crossed":
        reserve_crossed_seeds(output_root, seeds, days)
    elif role == "train_sensor_crossed":
        reserve_sensor_crossed_seeds(output_root, seeds, days)
    _install_source(source_root)
    from retrofit_sim import generate as G
    from retrofit_sim.dataset import asset_row, features_1min, ms_to_iso, snapshots_for
    templates = training_templates(source_root)
    dest = output_root / role
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "manifest.json").exists():
        raise FileExistsError(f"Completed {role} output already exists; choose a fresh output root")
    features, contexts, assets, events, runs, snapshots = [], [], [], [], [], []
    started = time.monotonic()
    for seed in seeds:
        track_count = (len(BALANCED_TRACKS) if role == "train_balanced" else
                       len(CROSSED_STRATA) * len(CROSSED_CLASSES) if role == "train_crossed" else
                       len(SENSOR_CROSSED_STRATA) * len(SENSOR_CROSSED_TARGETS) if role == "train_sensor_crossed" else len(templates))
        for index in range(track_count):
            tick = time.monotonic()
            a = (balanced_asset(templates, seed, index, days) if role == "train_balanced" else
                 crossed_asset(templates, seed, index, days) if role == "train_crossed" else
                 sensor_crossed_asset(templates, seed, index, days) if role == "train_sensor_crossed" else
                 randomized_asset(templates[index], seed, index, role, days))
            res = G.simulate_asset(a, seed, days, want_detectability=True)
            # The only feature-generation route is actual observed telemetry.
            f = features_1min(G.telemetry_frame(res))
            f["available_at_utc"] = f["window_start_utc"] + pd.Timedelta(minutes=1)
            features.append(f)
            contexts.append(G.context_frame(res))
            snapshots.extend(snapshots_for(res, interval_s=4 * 3600))
            metadata = asset_row(res["acfg"], res["specs"], res["acfg"]["limits"], days)
            metadata["template_asset_id"] = a["template_asset_id"]
            assets.append(metadata)
            ev = pd.DataFrame(G.events_for(res))
            if not ev.empty:
                ev["split"] = role
                events.append(ms_to_iso(ev, EVENT_TIME_COLS))
            # Train/dev plans are audit evidence. Holdout plans stay sealed in
            # this generation process and are not written alongside features.
            record = {"asset_id": a["asset_id"], "template_asset_id": a["template_asset_id"],
                      "seed": seed, "minutes": len(f), "seconds": round(time.monotonic() - tick, 3),
                      "instrumentation_tier": a["instrumentation_tier"],
                      "rating_scale_flow": a["rating_scale_flow"], "rating_scale_head": a["rating_scale_head"]}
            if role != "holdout":
                record.update(scenario_kind=a["scenario_kind"], scenario=a["scenario"],
                              event_classes=ev["fault_type"].value_counts().to_dict() if not ev.empty else {})
            runs.append(record)
            log(f"{role}: {a['asset_id']} {len(f)} observed minutes in {record['seconds']:.2f}s", flush=True)
            del res
    pd.concat(features, ignore_index=True).to_parquet(dest / "features_1min.parquet", index=False)
    pd.concat(contexts, ignore_index=True).to_csv(dest / "operating_context.csv", index=False)
    pd.DataFrame(assets).to_csv(dest / "assets.csv", index=False)
    all_events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    all_events.to_csv(dest / "events.csv", index=False)
    pd.DataFrame(snapshots).to_parquet(dest / "vibration_snapshots.parquet", index=False)
    manifest = {"role": role, "seeds": list(seeds), "days": days, "runs": runs,
                "observed_minute_rows": sum(len(f) for f in features),
                "asset_count": len(assets), "total_seconds": round(time.monotonic() - started, 3),
                "source_config_sha256": _sha256(source_root / "configs/fleet_public.yaml"),
                "generator_sha256": _sha256(Path(__file__)), "reserved_seed_manifest_sha256": _sha256(output_root / "seed_manifest.json"),
                "availability": "features are emitted at available_at_utc = window_start_utc + 60s",
                "inference_inputs": ["features_1min.parquet", "operating_context.csv", "assets.csv", "vibration_snapshots.parquet"],
                "snapshot_count": len(snapshots),
                "snapshot_caveat": "Published snapshots_for waveform route includes simulated measurement variation and missing gateways, but does not apply channel sensor-fault corruption; waveform use should be ablated.",
                "target_only": ["events.csv"], "latent_or_truth_exports": [],
                "artifacts": {name: _sha256(dest / name) for name in
                              ("features_1min.parquet", "operating_context.csv", "assets.csv", "events.csv", "vibration_snapshots.parquet")}}
    if role != "holdout":
        manifest["event_counts"] = all_events.fault_type.value_counts().to_dict() if not all_events.empty else {}
        manifest["observable_event_counts"] = all_events.loc[all_events.observable, "fault_type"].value_counts().to_dict() if not all_events.empty else {}
    else:
        manifest["frozen_pipeline_manifest_sha256"] = _sha256(Path(freeze_manifest))
        manifest["holdout_labels_inspected"] = False
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Wrote {dest}: {manifest['observed_minute_rows']} observed minutes; {manifest['total_seconds']}s", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--role", choices=(*ROLE_SEEDS, "train_balanced", "train_crossed", "train_sensor_crossed"), default="train")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--days", type=float, default=3)
    p.add_argument("--freeze-manifest", type=Path)
    p.add_argument("--reserve-only", action="store_true")
    args = p.parse_args()
    if args.reserve_only:
        reserve_seeds(args.output)
    else:
        generate_role(args.source, args.output, args.role, args.seeds, args.days, args.freeze_manifest)


if __name__ == "__main__":
    main()
