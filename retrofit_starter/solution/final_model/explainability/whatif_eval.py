"""Run retrofit_sim.what_if for the packet's sandbox_action (advice measurement only)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .sandbox_map import hypothesis_for


def evaluate_what_if(packet: dict[str, Any], data_dir: Path, challenge_root: Path | None = None) -> dict[str, Any]:
    root = Path(challenge_root) if challenge_root else Path(data_dir).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from retrofit_sim.sandbox import what_if

    action = str(packet.get("sandbox_action") or "inspect")
    hyp = hypothesis_for(str(packet.get("fault_type") or ""), action)
    result = what_if(
        str(data_dir),
        str(packet["asset_id"]),
        str(packet["alert_time_utc"]),
        action=action,
        value=None,
        hypothesis=hyp or None,
        root=str(root),
    )
    return {
        "sandbox_action": action,
        "hypothesis": hyp,
        "result": result,
    }
