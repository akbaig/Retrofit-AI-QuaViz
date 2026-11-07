"""Online adapter: raw 1 s telemetry (replay/JSONL) → 1-min features → frozen best."""
from .runner import OnlineBest

__all__ = ["OnlineBest"]
