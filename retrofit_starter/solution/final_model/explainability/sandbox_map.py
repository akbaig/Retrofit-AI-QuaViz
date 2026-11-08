"""Map detector fault_type → retrofit_sim.what_if action name."""
from __future__ import annotations

from typing import Any

# Sandbox INTERVENTIONS keys only.
SANDBOX_ACTIONS = {
    "SUCTION_STRAINER_CLOGGING": "clean_strainer",
    "BEARING_WEAR": "replace_bearing",
    "MISALIGNMENT": "realign",
    "DISCHARGE_RESTRICTION": "inspect",
    "IMBALANCE": "inspect",
    "IMPELLER_WEAR": "inspect",
    "AIR_ENTRAINMENT": "inspect",
    "SENSOR_DRIFT": "inspect",
    "SENSOR_STUCK": "inspect",
    "SENSOR_CALIBRATION_SHIFT": "inspect",
    "SENSOR_NOISE_SPIKES": "inspect",
    "COMMUNICATION_LOSS": "inspect",
}


def sandbox_action_for(fault_type: str, packet: dict[str, Any] | None = None) -> str:
    if fault_type == "CAVITATION":
        # Prefer strainer clean when suction/clog-style residuals dominate.
        names = []
        if packet:
            for key in ("top_residuals", "physics"):
                for row in packet.get(key) or []:
                    names.append(str(row.get("name", "")).lower())
        joined = " ".join(names)
        if any(tok in joined for tok in ("suction", "clog", "strainer", "npsh", "flow")):
            return "clean_strainer"
        return "inspect"
    return SANDBOX_ACTIONS.get(fault_type, "inspect")


def hypothesis_for(fault_type: str, sandbox_action: str) -> dict[str, float]:
    """Mild degraded state so what_if has something to improve (illustrative)."""
    if sandbox_action == "clean_strainer" or fault_type in ("SUCTION_STRAINER_CLOGGING", "CAVITATION"):
        return {"clog": 3.0}
    if sandbox_action == "replace_bearing" or fault_type == "BEARING_WEAR":
        return {"bearing_wear": 0.55}
    if sandbox_action == "realign" or fault_type == "MISALIGNMENT":
        return {"misalignment": 0.45}
    return {}
