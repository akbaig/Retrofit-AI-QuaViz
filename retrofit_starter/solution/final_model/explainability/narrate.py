"""Turn an explain packet into operator narration via the configured LLM."""
from __future__ import annotations

from typing import Any

from .llm import LLM, Narrative, get_llm
from .prompts import SYSTEM_PROMPT


def narrate_alert(packet: dict[str, Any], llm: LLM | None = None) -> Narrative:
    client = llm or get_llm()
    narrative = client.narrate(SYSTEM_PROMPT, packet)
    # Hard guarantee: model never overrides detector fault_type.
    narrative.fault_type = str(packet.get("fault_type") or narrative.fault_type)
    return narrative
