"""Randomized, fully labeled pump fleets generated with the challenge simulator.

Each seed produces one build folder with the same tables as the official data (``public/``) plus full labels and
true states (``private/``). Every seed covers all profile/control combinations at all three instrumentation tiers,
with random ratings, season, missing optional sensors and random fault plans, including the compound incidents the
challenge documents (bearing repair -> misalignment, planned work -> misalignment, restriction repair -> transmitter
offset, low suction level -> air entrainment, storm -> air entrainment, severe clogging -> cavitation, and a drifting
suction transmitter that masks a clogging strainer).

    python -m quaviz_detector.simfleet --seeds 101-110 --out build/sim --workers 4
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .paths import BUILD, PUBLIC_FLEET

COMBOS = [("SHIP_SW_COOLING", "VFD_TEMPERATURE"), ("SHIP_SW_COOLING", "FIXED_SPEED"), ("SHIP_LT_FRESHWATER", "FIXED_SPEED"),
          ("INDUSTRIAL_CW", "VFD_PRESSURE"), ("INDUSTRIAL_CW", "FIXED_SPEED"), ("PROCESS_TRANSFER", "FLOW_VALVE")]
TIERS = ["T0_LEGACY", "T1_RETROFIT", "T2_FULL"]
SENSOR_TAGS = ["discharge_pressure_kpa", "suction_pressure_kpa", "bearing_temp_c", "motor_current_a", "flow_m3_h"]
MACHINE_SINGLES = ["BEARING_WEAR", "SUCTION_STRAINER_CLOGGING", "DISCHARGE_RESTRICTION", "MISALIGNMENT", "IMBALANCE",
                   "IMPELLER_WEAR", "AIR_ENTRAINMENT"]
SENSOR_SINGLES = ["SENSOR_DRIFT", "SENSOR_STUCK", "SENSOR_CALIBRATION_SHIFT", "SENSOR_NOISE_SPIKES", "COMMUNICATION_LOSS"]
COMPOUNDS = ["bearing_repair_misalignment", "maintenance_misalignment", "restriction_repair_offset", "air_entrainment",
             "masked_clogging", "clogging_cavitation"]
PLANNED_WORK = [("MECHANICAL_SEAL_REPLACEMENT", "Mechanical seal replaced (planned). Coupling refitted, pump restarted."),
                ("MOTOR_OVERHAUL", "Motor overhauled (planned). Coupling refitted, pump restarted.")]
DAYS = 7.0
MIN_GAP_H = 8.0          # independent faults start at least this far apart (the simulator adds a few hours of onset jitter)


class _Rng:
    def __init__(self, seed):
        self.g = np.random.default_rng(seed)

    def u(self, a, b, nd=4):
        return round(float(self.g.uniform(a, b)), nd)

    def p(self, prob):
        return bool(self.g.random() < prob)

    def pick(self, seq):
        return seq[int(self.g.integers(0, len(seq)))]

    def sign(self):
        return 1.0 if self.g.random() < 0.5 else -1.0


def _motor_size(r: _Rng, p_shaft: float) -> float:
    """Standard motor with a random margin, kept within the package's plausibility rule (motor < 2 x shaft power)."""
    from retrofit_sim.stress import MOTOR_STEPS
    for margin in (r.u(1.12, 1.4), 1.12):
        m = next((s for s in MOTOR_STEPS if s >= margin * p_shaft), None)
        if m is not None and m < 2.0 * p_shaft:
            return float(m)
    raise ValueError(f"no standard motor fits shaft power {p_shaft:.1f} kW")


def _asset(r: _Rng, asset_id: str, prof: str, ctrl: str, tier: str, profiles: dict) -> dict:
    from retrofit_sim.fleet import rated_current_a
    from retrofit_sim.sensors import TIERS as TIER_CHANNELS
    from retrofit_sim.stress import RANGES
    (q0, q1), (h0, h1), rpms, (e0, e1) = RANGES[prof]
    q = float(np.round(r.u(q0, q1), -1 if q1 > 100 else 0))
    h = r.u(h0, h1, 1)
    eta = r.u(e0, e1, 3)
    p_shaft = 9.81 * q / 3600 * h / eta
    motor = _motor_size(r, p_shaft)
    month = int(r.g.integers(1, 13))
    a = {
        "asset_id": asset_id, "split": "train", "site": "Simulated site (fictional)",
        "system": f"{prof.replace('_', ' ').title()} pump ({ctrl})", "profile": prof, "control": ctrl,
        "start_utc": f"{2025 if month >= 4 else 2026}-{month:02d}-{int(r.g.integers(1, 21)):02d}T00:00:00",
        "commissioned_year": int(r.g.integers(1995, 2022)), "instrumentation_tier": tier,
        "q_bep_m3h": q, "h_bep_m": h, "eta_bep": eta, "npshr_bep_m": r.u(2.6, 4.8, 1),
        "q_design_m3h": float(round(r.u(0.85, 0.98) * q)), "h_design_m": round(r.u(0.93, 0.99) * h, 1),
        "motor_kw": motor, "motor_eff": round(0.90 + 0.05 * min(1.0, motor / 75), 3), "motor_pf": r.u(0.85, 0.87, 2),
        "rated_rpm": float(r.pick(list(rpms))),
        "vib_base_mm_s": r.u(0.9, 1.7, 2), "unbalance_base_mm_s": r.u(0.5, 0.8, 2), "hydraulic_vib_mm_s": r.u(0.9, 1.3, 2),
        "cavitation_vib_mm_s": r.u(2.5, 3.5, 2), "accel_base_g": r.u(0.13, 0.25, 3),
    }
    if ctrl == "VFD_PRESSURE":
        a["dp_setpoint_m"] = round(r.u(0.80, 0.88) * h, 1)
    if ctrl == "VFD_TEMPERATURE":
        a["r_min"] = float(r.pick([0.5, 0.55, 0.6]))
    if r.p(0.15):
        optional = [c for c in TIER_CHANNELS[tier] if c not in TIER_CHANNELS["T0_LEGACY"]]
        if optional:
            a["missing_channels"] = [r.pick(optional)]
    have = [t for t in SENSOR_TAGS if t in TIER_CHANNELS[tier] and t not in a.get("missing_channels", [])]
    a["scenario"], ctx = _scenario(r, a, have, rated_current_a({**a, "voltage_v": profiles[prof]["voltage_v"]}))
    if prof == "PROCESS_TRANSFER":
        ctx["process_temp_c"] = r.u(57, 65, 1)
    if ctx:
        a["context"] = ctx
    return a


def _magnitudes(a, i_r):
    q = a["q_bep_m3h"]
    return {"discharge_pressure_kpa": {"step": (8, 25), "drift": (0.15, 0.6), "spike": (40, 80)},
            "suction_pressure_kpa": {"step": (4, 12), "drift": (0.1, 0.4), "spike": (15, 30)},
            "bearing_temp_c": {"step": (3, 8), "drift": (0.1, 0.4), "spike": (6, 12)},
            "motor_current_a": {"step": (0.03 * i_r, 0.08 * i_r), "drift": (0.001 * i_r, 0.004 * i_r), "spike": (0.15 * i_r, 0.3 * i_r)},
            "flow_m3_h": {"step": (0.05 * q, 0.12 * q), "drift": (0.002 * q, 0.006 * q), "spike": (0.2 * q, 0.4 * q)}}


def _scenario(r: _Rng, a: dict, have: list, i_r: float):
    H = DAYS * 24
    S, ctx, used = [], {}, set()
    mag = _magnitudes(a, i_r)

    def add(e):
        S.append(e)
        used.add((e["type"], e.get("sensor_tag")))
        return len(S)

    def end_after(onset, lo, hi):
        return round(min(H - 1.0, max(onset, 0.0) + r.u(lo, hi)), 1)

    def clear(onset):
        return all(abs(max(onset, 0.0) - max(e["onset_h"], 0.0)) >= MIN_GAP_H for e in S)

    def spaced(make, tries=12):
        """A fault from ``make`` whose onset keeps clear of every onset already planned; None if none fits."""
        for _ in range(tries):
            e = make()
            if clear(e["onset_h"]):
                return e
        return None

    def free_onset(lo, hi, tries=12):
        for _ in range(tries):
            onset = r.u(lo, hi, 1)
            if clear(onset):
                return onset
        return None

    def bearing(pre=None, force_end=False):
        # official configs: 3 of 5 bearing faults present at window start, the others starting at 10 h and 30 h
        pre = r.p(0.6) if pre is None else pre
        onset = -r.u(30, 120, 1) if pre else r.u(4, 40, 1)
        e = {"type": "BEARING_WEAR", "onset_h": onset, "params": {"rate_per_h": r.u(0.008, 0.014), "t_fail_h": r.u(150, 420, 1)}}
        if force_end or r.p(0.4):
            # official repairs come late in the week (about 130 h); earlier ones mostly cut the wear off before it shows
            e["end_h"] = round(min(H - 30, max(onset, 0) + r.u(80, 140)), 1)
            e["repair_h"] = r.u(5, 7, 1)
        return e

    def clogging(severe=False):
        onset = r.u(5, 0.7 * H, 1)
        e = {"type": "SUCTION_STRAINER_CLOGGING", "onset_h": onset,
             "params": {"rate_per_h": r.u(0.08, 0.14) if severe else r.u(0.03, 0.08),
                        "c_max": r.u(5, 7, 2) if severe else r.u(2.5, 4, 2), "shape": "linear"}}
        if r.p(0.6):
            e["end_h"], e["repair_h"] = end_after(onset, 25, 60), r.u(1, 2, 1)
        return e

    def restriction(force_end=False):
        onset = r.u(5, 0.7 * H, 1)
        e = {"type": "DISCHARGE_RESTRICTION", "onset_h": onset,
             "params": {"rate_per_h": float(r.pick([0.03, 0.1, 1.5, 3.0])), "extra_max": r.u(0.5, 1.3, 2), "shape": "exp"}}
        if force_end or r.p(0.6):
            e["end_h"], e["repair_h"] = round(min(H - 40, onset + r.u(20, 55)), 1), r.u(1, 4, 1)
        return e

    def misalignment():
        onset = r.u(5, 0.8 * H, 1)
        e = {"type": "MISALIGNMENT", "onset_h": onset, "params": {"level": r.u(0.35, 0.65, 3), "growth_per_h": r.u(0, 0.004)}}
        if r.p(0.4):
            e["end_h"], e["repair_h"] = end_after(onset, 30, 70), 3.0
        return e

    def imbalance():
        return {"type": "IMBALANCE", "onset_h": r.u(5, 0.6 * H, 1),
                "params": {"rate_per_h": r.u(0.006, 0.015), "u_max": r.u(0.7, 1.5, 2), "shape": "linear"}}

    def impeller():
        return {"type": "IMPELLER_WEAR", "onset_h": -r.u(100, 600, 0), "jitter_h": 0,
                "params": {"rate_per_h": r.u(3e-5, 8e-5, 6), "iw0": r.u(0.03, 0.08, 3), "iw_max": 0.12}}

    def air():
        onset = free_onset(10, 0.85 * H)
        if onset is None:
            return
        dur = r.u(3, 8, 1)
        if a["profile"].startswith("SHIP"):
            ctx.update({"storm_at_h": round(max(0.0, onset - r.u(2, 6)), 1), "hs_mean": r.u(1.2, 1.8, 2)})
            add({"type": "AIR_ENTRAINMENT", "onset_h": onset, "end_h": round(min(H - 1, onset + dur), 1),
                 "params": {"alpha_max": r.u(0.02, 0.04)}})
        else:
            li = add({"type": "SUCTION_LEVEL_DROP", "onset_h": onset, "end_h": round(min(H - 1, onset + dur), 1),
                      "params": {"drop_m": r.u(1.6, 2.2, 2), "ramp_h": r.u(1.0, 2.0, 2)}})
            add({"type": "AIR_ENTRAINMENT", "onset_h": round(onset + 1, 1), "end_h": round(min(H - 1, onset + dur), 1),
                 "root_cause": li, "align": "onset", "params": {"alpha_max": r.u(0.02, 0.035)}})

    def sensor(typ, tag=None):
        onset = r.u(3, 0.8 * H, 1)
        if typ == "COMMUNICATION_LOSS":
            return {"type": typ, "sensor_tag": "GATEWAY", "onset_h": onset, "end_h": end_after(onset, 0.5, 3), "jitter_h": 5}
        m = mag[tag]
        e = {"type": typ, "sensor_tag": tag, "onset_h": onset}
        if typ == "SENSOR_DRIFT":
            e["params"] = {"drift_per_h": round(r.sign() * r.u(*m["drift"]), 4)}
            if r.p(0.6):
                e["end_h"] = end_after(onset, 30, 90)
        elif typ == "SENSOR_CALIBRATION_SHIFT":
            e["params"] = {"step": round(r.sign() * r.u(*m["step"]), 3)}
            if r.p(0.7):
                e["end_h"] = end_after(onset, 20, 60)
        elif typ == "SENSOR_STUCK":
            e["end_h"] = end_after(onset, 4, 26)
        elif typ == "SENSOR_NOISE_SPIKES":
            e["end_h"] = end_after(onset, 5, 14)
            e["params"] = {"noise_mult": r.u(3, 6, 2), "spike_rate": r.u(0.002, 0.005), "spike_size": round(r.u(*m["spike"]), 2)}
        return e

    def compound(kind):
        if kind == "bearing_repair_misalignment":
            b = spaced(lambda: bearing(force_end=True))
            if b is None:
                return
            bi = add(b)
            add({"type": "MISALIGNMENT", "onset_h": round(b["end_h"] + b["repair_h"] + 0.5, 1), "root_cause": bi, "jitter_h": 0,
                 "params": {"level": r.u(0.4, 0.65, 3), "growth_per_h": r.u(0, 0.004)}})
        elif kind == "maintenance_misalignment":
            wt, note = r.pick(PLANNED_WORK)
            onset = free_onset(2, 100)
            if onset is None:
                return
            pi = add({"type": "PLANNED_MAINTENANCE", "onset_h": onset, "repair_h": r.u(2, 4, 1), "jitter_h": 1,
                      "params": {"work_type": wt, "note": note}})
            add({"type": "MISALIGNMENT", "onset_h": onset, "root_cause": pi, "jitter_h": 0,
                 "params": {"level": r.u(0.3, 0.5, 3), "growth_per_h": r.u(0, 0.003)}})
        elif kind == "restriction_repair_offset":
            x = spaced(lambda: restriction(force_end=True))
            if x is None:
                return
            xi = add(x)
            onset = round(x["end_h"] + x["repair_h"] + 0.5, 1)
            add({"type": "SENSOR_CALIBRATION_SHIFT", "sensor_tag": "discharge_pressure_kpa", "onset_h": onset,
                 "end_h": round(min(H - 1, onset + r.u(30, 50)), 1), "root_cause": xi, "jitter_h": 0,
                 "params": {"step": round(r.sign() * r.u(*mag["discharge_pressure_kpa"]["step"]), 3)}})
        elif kind == "air_entrainment":
            air()
        elif kind == "masked_clogging":
            # the drift usually runs before the strainer starts to clog, so the suction reading never shows the drop
            c = spaced(clogging)
            if c is None:
                return
            add(c)
            lead = r.u(MIN_GAP_H, 30) if r.p(0.7) else -r.u(MIN_GAP_H, 16)
            add({"type": "SENSOR_DRIFT", "sensor_tag": "suction_pressure_kpa", "onset_h": round(c["onset_h"] - lead, 1),
                 "params": {"drift_per_h": r.u(0.2, 0.5)}})
        elif kind == "clogging_cavitation":
            c = spaced(lambda: clogging(severe=True))
            if c is not None:
                add(c)

    allowed = [k for k in COMPOUNDS if k != "masked_clogging" or "suction_pressure_kpa" in have]
    n_comp = int(r.g.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))
    for kind in r.g.choice(allowed, size=n_comp, replace=False):
        compound(str(kind))
    makers = {"BEARING_WEAR": bearing, "SUCTION_STRAINER_CLOGGING": clogging, "DISCHARGE_RESTRICTION": restriction,
              "MISALIGNMENT": misalignment, "IMBALANCE": imbalance, "IMPELLER_WEAR": impeller}
    for _ in range(int(r.g.integers(1, 5))):
        if r.p(0.55):
            typ = r.pick(MACHINE_SINGLES)
            if any(u[0] == typ for u in used):
                continue
            if typ == "BEARING_WEAR" and a["instrumentation_tier"] == "T0_LEGACY" and r.p(0.7):
                continue                      # never observable without vibration or bearing temperature; keep a few
            if typ == "AIR_ENTRAINMENT":
                air()
            elif (e := spaced(makers[typ])) is not None:
                add(e)
        else:
            typ = r.pick(SENSOR_SINGLES)
            tag = "GATEWAY" if typ == "COMMUNICATION_LOSS" else (r.pick(have) if have else None)
            if tag is None or (typ, tag) in used:
                continue
            if (e := spaced(lambda: sensor(typ, tag if tag != "GATEWAY" else None))) is not None:
                add(e)
    if not any(e["type"] in MACHINE_SINGLES for e in S):
        make = makers[r.pick(["BEARING_WEAR", "SUCTION_STRAINER_CLOGGING", "DISCHARGE_RESTRICTION"])]
        add(spaced(make) or make())
    return S, ctx


def fleet_config(seed: int, assets_per_combo: int = 1) -> dict:
    public = yaml.safe_load(PUBLIC_FLEET.read_text(encoding="utf-8"))
    r = _Rng([int(seed), 2026])
    cells = [(p, c, t) for p, c in COMBOS for t in TIERS] * assets_per_combo
    order = r.g.permutation(len(cells))
    assets = [_asset(r, f"SIM{seed:04d}-{k + 1:02d}", *cells[j], public["profiles"]) for k, j in enumerate(order)]
    return {"dataset_version": f"quaviz-sim-{seed}", "public_seed": int(seed), "profiles": public["profiles"], "assets": assets}


def generate(seed: int, out_root=BUILD / "sim", snapshots: bool = False, assets_per_combo: int = 1,
             keep_telemetry: bool = False) -> Path:
    import shutil

    from retrofit_sim.dataset import build
    out = Path(out_root) / f"s{seed:04d}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = fleet_config(seed, assets_per_combo)
    cfg_path = out / "fleet.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    t0 = time.time()
    # retrofit_sim.records.alarms indexes one past the window end when an alarm is due on the very last second;
    # a different simulation seed (same fleet config) avoids it
    for attempt in range(3):
        sim_seed = seed + 100_000 * attempt
        try:
            build("full", sim_seed, str(out), profile="public", fleet_config=str(cfg_path), days=DAYS, snapshots=snapshots,
                  log=lambda *_: None)
            break
        except IndexError:
            if attempt == 2:
                raise
    if not keep_telemetry:
        shutil.rmtree(out / "public" / "telemetry", ignore_errors=True)
    (out / "generation.json").write_text(json.dumps({"seed": seed, "sim_seed": sim_seed, "assets": len(cfg["assets"]),
                                                     "seconds": round(time.time() - t0, 1)}), encoding="utf-8")
    return out


def _parse_seeds(spec: list[str]) -> list[int]:
    out = []
    for s in spec:
        if "-" in s:
            a, b = s.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(s))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m quaviz.simfleet", description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", nargs="+", default=["101"], help="Seeds or ranges, e.g. 101-110 120")
    p.add_argument("--out", default=str(BUILD / "sim"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--assets-per-combo", type=int, default=1, help="Assets per profile/control/tier cell (18 cells).")
    p.add_argument("--snapshots", action="store_true", help="Also write vibration waveform snapshots (slower, larger).")
    p.add_argument("--keep-telemetry", action="store_true", help="Keep the 1-s telemetry (about 95 MB per seed).")
    a = p.parse_args(argv)
    seeds = _parse_seeds(a.seeds)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(generate, s, a.out, a.snapshots, a.assets_per_combo, a.keep_telemetry): s for s in seeds}
        for f, s in futs.items():
            try:
                print(f"seed {s}: {f.result()}")
            except Exception as e:      # one failing seed must not stop the others
                print(f"seed {s}: FAILED {type(e).__name__}: {e}")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
