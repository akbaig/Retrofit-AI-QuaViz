"""Shared JSON narrative parsing for all providers."""
from __future__ import annotations

import json
import re
from typing import Any

from .base import Narrative


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(x) for x in value if str(x).strip()]


def parse_narrative(text: str, packet: dict[str, Any], provider: str, model: str) -> Narrative:
    """Extract Narrative from model text; tolerate fenced JSON."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON")
        data = json.loads(match.group(0))
    fault = str(packet.get("fault_type") or "")
    return Narrative(
        summary=str(data.get("summary") or "").strip(),
        likely_cause=str(data.get("likely_cause") or "").strip(),
        what_to_check=_as_list(data.get("what_to_check")),
        do_not_do=_as_list(data.get("do_not_do")),
        uncertainty_note=str(data.get("uncertainty_note") or "").strip(),
        provider=provider,
        model=model,
        fault_type=fault,
    )
