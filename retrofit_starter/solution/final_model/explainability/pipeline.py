"""End-to-end: packet → LLM narrative → optional what-if."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .llm import LLM, get_llm
from .narrate import narrate_alert
from .packet import explain_alert, load_frozen
from .whatif_eval import evaluate_what_if


def explain_and_advise(
    *,
    data_dir: Path,
    asset_id: str,
    model_path: Path,
    policy_path: Path,
    alert_index: int = 0,
    llm: LLM | None = None,
    run_what_if: bool = True,
    challenge_root: Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    model, policy = load_frozen(model_path, policy_path)
    assets = pd.read_csv(data_dir / "assets.csv")
    asset = assets[assets.asset_id == asset_id]
    if asset.empty:
        raise ValueError(f"Unknown asset_id {asset_id}")
    asset = asset.iloc[0]
    features = pd.read_parquet(data_dir / "features_1min.parquet", filters=[("asset_id", "==", asset_id)])
    ctx_path, snap_path = data_dir / "operating_context.csv", data_dir / "vibration_snapshots.parquet"
    context = pd.read_csv(ctx_path) if ctx_path.exists() else None
    snapshots = (
        pd.read_parquet(snap_path, filters=[("asset_id", "==", asset_id)]) if snap_path.exists() else None
    )
    explained = explain_alert(
        model=model, policy=policy, features=features, asset=asset,
        context=context, snapshots=snapshots, alert_index=alert_index,
    )
    packet = explained["packet"]
    narrative = narrate_alert(packet, llm=llm or get_llm())
    out: dict[str, Any] = {
        "alert": explained["alert"],
        "packet": packet,
        "narrative": narrative.to_dict(),
    }
    if run_what_if:
        out["what_if"] = evaluate_what_if(packet, data_dir, challenge_root=challenge_root)
    return out
