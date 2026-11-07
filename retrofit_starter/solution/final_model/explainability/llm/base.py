"""Provider-agnostic LLM contract for operator narration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Narrative:
    summary: str
    likely_cause: str
    what_to_check: list[str] = field(default_factory=list)
    do_not_do: list[str] = field(default_factory=list)
    uncertainty_note: str = ""
    provider: str = ""
    model: str = ""
    fault_type: str = ""  # echoed from packet; never overridden by the model claim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLM(Protocol):
    """Any provider that can turn an explain packet into a Narrative."""

    name: str
    model: str

    def narrate(self, system_prompt: str, packet: dict[str, Any]) -> Narrative:
        """Return structured operator narration. Must not invent fault_type."""
        ...
