"""Live/near-live path: replay or JSONL → 1-min features → frozen best alerts.

Examples:
  # Internal replay source (no shell pipe needed)
  python -m deploy --asset PUMP-08 --limit 7200 --speed 0

  # Pipe from the challenge toolkit
  python -m retrofit_sim replay --data data/full --asset PUMP-08 --limit 7200 \\
    | python -m deploy --asset PUMP-08 --stdin

  # Also narrate the first new alert (optional LLM)
  python -m deploy --asset PUMP-08 --limit 20000 --explain --provider template
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
if str(FINAL) not in sys.path:
    sys.path.insert(0, str(FINAL))

from paths import SOURCE  # noqa: E402

from deploy.aggregate import asset_id_from_topic  # noqa: E402
from deploy.buffer import MinuteBuffer  # noqa: E402
from deploy.runner import OnlineBest  # noqa: E402


def _iter_stdin():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _iter_replay(data: Path, asset: str, start, end, limit, speed: float):
    sys.path.insert(0, str(SOURCE / "src"))
    from retrofit_sim.replay import messages
    import time

    prev = None
    n = 0
    for m in messages(str(data), asset, start, end, limit):
        if speed > 0:
            ts = pd.Timestamp(m["payload"]["timestamp_utc"])
            if prev is not None:
                time.sleep(max(0.0, (ts - prev).total_seconds() / speed))
            prev = ts
        yield m
        n += 1
    print(f"replayed {n} messages", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=SOURCE / "data" / "full")
    p.add_argument("--asset", required=True)
    p.add_argument("--model", type=Path, default=FINAL / "model.joblib")
    p.add_argument("--policy", type=Path, default=FINAL / "policy.json")
    p.add_argument("--stdin", action="store_true", help="Read retrofit_sim replay JSONL from stdin")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", type=int, default=None, help="Max raw messages (internal replay)")
    p.add_argument("--speed", type=float, default=0.0, help="Internal replay speed (0=asap)")
    p.add_argument("--infer-every", type=int, default=5,
                   help="Run best every N completed minutes (0 = only at end of stream)")
    p.add_argument("--explain", action="store_true", help="Narrate each new alert (uses LLM_PROVIDER)")
    p.add_argument("--provider", choices=["gemini", "ollama", "template"], default=None)
    p.add_argument("--what-if", action="store_true", help="With --explain, also run sandbox what-if")
    p.add_argument("--alerts-out", type=Path, default=None)
    a = p.parse_args(argv)

    online = OnlineBest(
        model_path=a.model, policy_path=a.policy, data_dir=a.data, asset_id=a.asset,
    )
    buf = MinuteBuffer(asset_id=a.asset)
    all_alerts = []
    minutes = 0

    stream = _iter_stdin() if a.stdin else _iter_replay(
        a.data, a.asset, a.start, a.end, a.limit, a.speed,
    )

    for msg in stream:
        if isinstance(msg, dict) and "payload" in msg:
            topic, payload = msg.get("topic"), msg["payload"]
            aid = asset_id_from_topic(topic) or payload.get("asset_id") or a.asset
            if aid != a.asset:
                continue
            payload = {**payload, "asset_id": aid}
        else:
            payload = msg
        for minute_frame in buf.push(payload):
            online.extend_features(minute_frame)
            minutes += 1
            if a.infer_every > 0 and minutes % a.infer_every == 0:
                _emit_fresh(online, all_alerts, a)

    for minute_frame in buf.flush():
        online.extend_features(minute_frame)
        minutes += 1
    # Always score once at end so --infer-every 0 and partial windows still alert.
    _emit_fresh(online, all_alerts, a)

    summary = {
        "type": "summary",
        "asset_id": a.asset,
        "minutes_aggregated": minutes,
        "feature_rows": int(len(online.features)),
        "alerts": len(all_alerts),
    }
    print(json.dumps(summary), flush=True)
    if a.alerts_out:
        pd.DataFrame(all_alerts).to_csv(a.alerts_out, index=False)
        print(f"wrote {a.alerts_out}", file=sys.stderr)
    return 0


def _emit_fresh(online: OnlineBest, all_alerts: list, args) -> None:
    fresh = online.infer_new_alerts()
    if not len(fresh):
        return
    for _, row in fresh.iterrows():
        rec = row.to_dict()
        print(json.dumps({"type": "alert", **rec}, default=str), flush=True)
        all_alerts.append(rec)
        if args.explain:
            _explain_one(rec, args)


def _explain_one(alert: dict, args) -> None:
    try:
        from explainability.llm.factory import get_llm, load_dotenv
        from explainability.pipeline import explain_and_advise
    except ImportError:
        print(json.dumps({"type": "explain_error", "error": "explainability package missing"}), flush=True)
        return
    load_dotenv()
    # Full card via frozen path at alert time (uses same data_dir assets/features path through pipeline).
    # For streaming we rebuild from the alert fields + a one-shot explain_and_advise on stored data is heavier;
    # use lightweight packet from alert evidence when features row is not re-fetched.
    try:
        card = explain_and_advise(
            data_dir=args.data,
            asset_id=alert["asset_id"],
            model_path=args.model,
            policy_path=args.policy,
            alert_index=0,
            llm=get_llm(args.provider),
            run_what_if=args.what_if,
            challenge_root=SOURCE,
        )
        # Prefer the matching alert time if present among regenerated alerts.
        print(json.dumps({"type": "explain", "alert_time_utc": alert["alert_time_utc"],
                          "narrative": card["narrative"],
                          "sandbox_action": card["packet"].get("sandbox_action"),
                          "what_if_accepted": (card.get("what_if") or {}).get("result", {}).get("accepted")},
                         default=str), flush=True)
    except Exception as exc:  # demo path: never kill the stream
        print(json.dumps({"type": "explain_error", "error": str(exc)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
