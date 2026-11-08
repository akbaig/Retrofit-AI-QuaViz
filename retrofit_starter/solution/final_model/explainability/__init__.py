"""Operator explainability on top of frozen best: packets, LLM narration, what-if."""
from .narrate import narrate_alert
from .packet import build_explain_packet, explain_alert
from .pipeline import explain_and_advise

__all__ = ["build_explain_packet", "explain_alert", "narrate_alert", "explain_and_advise"]
