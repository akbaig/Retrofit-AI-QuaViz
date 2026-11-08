"""Deterministic fallback when no remote/local LLM is available."""
from __future__ import annotations

from typing import Any

from .base import Narrative


class TemplateLLM:
    name = "template"
    model = "template"

    def narrate(self, system_prompt: str, packet: dict[str, Any]) -> Narrative:
        del system_prompt  # unused; keeps Protocol signature
        fault = str(packet.get("fault_type") or "UNKNOWN")
        conf = packet.get("confidence")
        conf_s = f"{float(conf):.2f}" if conf is not None else "n/a"
        residuals = packet.get("top_residuals") or []
        phys = packet.get("physics") or []
        bits = []
        for row in residuals[:3]:
            bits.append(f"{row['name']}={row['value']:+.2f}")
        for row in phys[:2]:
            bits.append(f"{row['name']}={row['value']:+.2f}")
        evidence = "; ".join(bits) if bits else packet.get("evidence") or "model score only"
        checks = [
            str(packet.get("recommended_action_text") or "Follow the recommended inspection."),
        ]
        if packet.get("event_class") == "SENSOR":
            checks.append("Compare the suspect channel with a redundant or portable instrument before acting on the process.")
            do_not = ["Do not shut down or overhaul the pump solely on a single-channel sensor anomaly."]
        else:
            checks.append("Confirm the operating mode (speed, demand, suction) matches the evidence window.")
            do_not = ["Do not issue drive or valve commands from this assistant; human approval required."]
        return Narrative(
            summary=f"{fault} flagged with model confidence {conf_s}. Leading evidence: {evidence}.",
            likely_cause=f"Detector class {fault} (event class {packet.get('event_class')}).",
            what_to_check=checks,
            do_not_do=do_not,
            uncertainty_note="Template narration only; confidence is the raw model score and is not calibrated.",
            provider=self.name,
            model=self.model,
            fault_type=fault,
        )
