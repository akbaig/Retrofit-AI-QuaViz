"""Batch explain + what-if over validation alerts.

Reads original_validation_predictions.csv (or re-infers), runs explain_and_advise
for each alert, and writes a results CSV plus optional JSON cards.

Fixed := accepted AND proposed.risk_proxy < baseline.risk_proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
if str(FINAL) not in sys.path:
    sys.path.insert(0, str(FINAL))

from paths import SOURCE  # noqa: E402

from explainability.llm.factory import get_llm, load_dotenv  # noqa: E402
from explainability.pipeline import explain_and_advise  # noqa: E402

RESULT_FIELDS = [
    "asset_id",
    "alert_index",
    "alert_time_utc",
    "fault_type",
    "event_class",
    "confidence",
    "sandbox_action",
    "accepted",
    "fixed",
    "baseline_risk_proxy",
    "proposed_risk_proxy",
    "delta_risk_proxy",
    "baseline_flow_m3h",
    "proposed_flow_m3h",
    "baseline_vib_rms_mm_s",
    "proposed_vib_rms_mm_s",
    "estimated_cost_eur",
    "rejection_reasons",
    "warnings",
    "hypothesis",
    "llm_provider",
    "llm_model",
    "narrative_fault_type",
    "fault_type_preserved",
    "narrative_summary",
]


def _risk(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot:
        return None
    v = snapshot.get("risk_proxy")
    return float(v) if v is not None else None


def _row_from_card(card: dict[str, Any], alert_index: int) -> dict[str, Any]:
    alert = card.get("alert") or {}
    packet = card.get("packet") or {}
    wi = card.get("what_if") or {}
    result = wi.get("result") or {}
    baseline = result.get("baseline") or {}
    proposed = result.get("proposed") or {}
    intervention = result.get("intervention") or {}
    br = _risk(baseline)
    pr = _risk(proposed)
    accepted = bool(result.get("accepted"))
    fixed = bool(accepted and br is not None and pr is not None and pr < br)
    reasons = result.get("rejection_reasons") or []
    warnings = result.get("warnings") or []
    narrative = card.get("narrative") or {}
    fault = alert.get("fault_type") or packet.get("fault_type")
    narr_fault = narrative.get("fault_type")
    return {
        "asset_id": alert.get("asset_id") or packet.get("asset_id"),
        "alert_index": alert_index,
        "alert_time_utc": alert.get("alert_time_utc") or packet.get("alert_time_utc"),
        "fault_type": fault,
        "event_class": alert.get("event_class") or packet.get("event_class"),
        "confidence": alert.get("confidence") or packet.get("confidence"),
        "sandbox_action": wi.get("sandbox_action") or packet.get("sandbox_action"),
        "accepted": accepted,
        "fixed": fixed,
        "baseline_risk_proxy": br,
        "proposed_risk_proxy": pr,
        "delta_risk_proxy": (None if br is None or pr is None else pr - br),
        "baseline_flow_m3h": baseline.get("flow_m3h"),
        "proposed_flow_m3h": proposed.get("flow_m3h"),
        "baseline_vib_rms_mm_s": baseline.get("vib_rms_mm_s"),
        "proposed_vib_rms_mm_s": proposed.get("vib_rms_mm_s"),
        "estimated_cost_eur": intervention.get("estimated_cost_eur"),
        "rejection_reasons": "; ".join(str(x) for x in reasons),
        "warnings": "; ".join(str(x) for x in warnings),
        "hypothesis": json.dumps(wi.get("hypothesis") or {}, sort_keys=True),
        "llm_provider": narrative.get("provider"),
        "llm_model": narrative.get("model"),
        "narrative_fault_type": narr_fault,
        "fault_type_preserved": bool(fault) and str(narr_fault) == str(fault),
        "narrative_summary": (narrative.get("summary") or "").replace("\n", " ").strip(),
    }


def _alert_plan(predictions_csv: Path) -> list[tuple[str, int]]:
    """(asset_id, alert_index) in file order, indices restart per asset."""
    rows: list[tuple[str, int]] = []
    counts: dict[str, int] = {}
    with predictions_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            asset = str(row["asset_id"])
            idx = counts.get(asset, 0)
            counts[asset] = idx + 1
            rows.append((asset, idx))
    return rows


def run_batch(
    *,
    data_dir: Path,
    model_path: Path,
    policy_path: Path,
    predictions_csv: Path,
    provider: str | None,
    run_what_if: bool,
    challenge_root: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = output_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    llm = get_llm(provider)
    plan = _alert_plan(predictions_csv)
    results: list[dict[str, Any]] = []

    for asset_id, alert_index in plan:
        print(f"[{len(results)+1}/{len(plan)}] {asset_id} alert_index={alert_index}", flush=True)
        card = explain_and_advise(
            data_dir=data_dir,
            asset_id=asset_id,
            model_path=model_path,
            policy_path=policy_path,
            alert_index=alert_index,
            llm=llm,
            run_what_if=run_what_if,
            challenge_root=challenge_root,
        )
        card_path = cards_dir / f"{asset_id}_{alert_index:02d}.json"
        card_path.write_text(json.dumps(card, indent=2, default=str) + "\n", encoding="utf-8")
        results.append(_row_from_card(card, alert_index))

    out_csv = output_dir / "whatif_validation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    providers = sorted({str(r.get("llm_provider") or "") for r in results})
    summary = {
        "n_alerts": len(results),
        "n_accepted": sum(1 for r in results if r["accepted"]),
        "n_fixed": sum(1 for r in results if r["fixed"]),
        "n_repair_actions": sum(
            1
            for r in results
            if r["sandbox_action"] in ("clean_strainer", "replace_bearing", "realign")
        ),
        "n_inspect": sum(1 for r in results if r["sandbox_action"] == "inspect"),
        "n_fault_type_preserved": sum(1 for r in results if r.get("fault_type_preserved")),
        "llm_providers_seen": providers,
        "provider": provider or "env_default",
        "predictions_csv": str(predictions_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv}")
    return out_csv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Batch validation explain + what-if")
    p.add_argument("--data", type=Path, default=SOURCE / "data" / "full")
    p.add_argument("--model", type=Path, default=FINAL / "model.joblib")
    p.add_argument("--policy", type=Path, default=FINAL / "policy.json")
    p.add_argument(
        "--predictions",
        type=Path,
        default=FINAL / "results" / "original_validation_predictions.csv",
    )
    p.add_argument("--provider", choices=["gemini", "ollama", "template"], default="gemini")
    p.add_argument("--no-what-if", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=FINAL / "results" / "whatif_validation",
    )
    a = p.parse_args(argv)
    # Fail loud if Gemini/Ollama cannot start (do not silently fall back to template).
    if a.provider in ("gemini", "ollama"):
        import os

        os.environ.setdefault("LLM_REQUIRE_PROVIDER", "1")
    run_batch(
        data_dir=a.data,
        model_path=a.model,
        policy_path=a.policy,
        predictions_csv=a.predictions,
        provider=a.provider,
        run_what_if=not a.no_what_if,
        challenge_root=SOURCE,
        output_dir=a.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
