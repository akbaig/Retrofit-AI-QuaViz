"""CLI: serve (MQTT + API) or stdin pipe demo."""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading

from .config import Settings
from .mqtt import IngestWorker, MqttSubscriber
from .paths import bootstrap_imports
from .pipeline import PipelineHub
from .store import EventStore

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quaviz_backend",
        description="QuaViz product backend: MQTT/stdin → aggregate → best → explain → API",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run MQTT subscriber + HTTP API")
    _add_common(serve)
    serve.add_argument("--mqtt-host", default=None, help="Broker host (or MQTT_HOST)")
    serve.add_argument("--mqtt-port", type=int, default=None)
    serve.add_argument("--mqtt-topic", default=None)
    serve.add_argument("--host", default=None, help="HTTP bind host")
    serve.add_argument("--port", type=int, default=None, help="HTTP port")
    serve.add_argument(
        "--stdin",
        action="store_true",
        help="Also read retrofit_sim replay JSONL from stdin (no broker needed)",
    )

    pipe = sub.add_parser("stdin", help="One-shot: read replay JSONL from stdin, score, exit")
    _add_common(pipe)
    return p


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--asset", action="append", dest="assets", default=None,
                   help="Restrict to asset id (repeatable). Default: any known asset.")
    p.add_argument("--infer-every", type=int, default=None)
    p.add_argument("--explain", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--provider", choices=["gemini", "ollama", "template"], default=None)
    p.add_argument("--what-if", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--data", default=None, help="Override data/full directory")
    p.add_argument("--model", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("-v", "--verbose", action="store_true")


def _settings_from_args(args) -> Settings:
    overrides = {}
    if args.assets:
        overrides["asset_ids"] = args.assets
    if getattr(args, "infer_every", None) is not None:
        overrides["infer_every"] = args.infer_every
    if args.explain is not None:
        overrides["explain"] = args.explain
    if args.provider is not None:
        overrides["llm_provider"] = args.provider
    if args.what_if is not None:
        overrides["what_if"] = args.what_if
    if getattr(args, "mqtt_host", None):
        overrides["mqtt_host"] = args.mqtt_host
    if getattr(args, "mqtt_port", None) is not None:
        overrides["mqtt_port"] = args.mqtt_port
    if getattr(args, "mqtt_topic", None):
        overrides["mqtt_topic"] = args.mqtt_topic
    if getattr(args, "host", None):
        overrides["http_host"] = args.host
    if getattr(args, "port", None) is not None:
        overrides["http_port"] = args.port

    s = Settings.from_env(**overrides)
    if args.data:
        from pathlib import Path
        s.data_dir = Path(args.data).expanduser().resolve()
    if args.model:
        from pathlib import Path
        s.model_path = Path(args.model).expanduser().resolve()
    if args.policy:
        from pathlib import Path
        s.policy_path = Path(args.policy).expanduser().resolve()
    return s


def cmd_serve(args) -> int:
    settings = _settings_from_args(args)
    bootstrap_imports(settings.challenge_dir)
    from .jobs import BackgroundJobs

    store = EventStore()
    hub = PipelineHub(settings, store)
    jobs = BackgroundJobs(hub)
    hub.attach_jobs(jobs)
    jobs.start()

    q: queue.Queue = queue.Queue(maxsize=100_000)
    worker = IngestWorker(hub, q, on_error=lambda e: store.add_error(e))
    worker.start()

    mqtt = None
    if settings.mqtt_host:
        mqtt = MqttSubscriber(
            host=settings.mqtt_host,
            port=settings.mqtt_port,
            topic=settings.mqtt_topic,
            client_id=settings.mqtt_client_id,
            out=q,
        )
        mqtt.start()
        log.info(
            "MQTT mode: %s:%s topic=%s",
            settings.mqtt_host,
            settings.mqtt_port,
            settings.mqtt_topic,
        )
    elif not args.stdin:
        log.warning("No --mqtt-host / MQTT_HOST; API only until you pass --stdin or set MQTT_HOST")

    if args.stdin:
        def _stdin_then_flush():
            import time

            _pump_stdin(q)
            while not q.empty():
                time.sleep(0.05)
            time.sleep(0.2)
            hub.flush_all()
            log.info("stdin exhausted; flushed pipelines")

        threading.Thread(target=_stdin_then_flush, name="stdin-pump", daemon=True).start()

    from .api import create_app
    import uvicorn

    app = create_app(hub, store)
    try:
        uvicorn.run(app, host=settings.http_host, port=settings.http_port, log_level="info")
    finally:
        if mqtt:
            mqtt.stop()
        worker.stop(flush=True)
        jobs.stop()
    return 0


def cmd_stdin(args) -> int:
    settings = _settings_from_args(args)
    bootstrap_imports(settings.challenge_dir)
    store = EventStore()
    hub = PipelineHub(settings, store)
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        hub.ingest_message(json.loads(line))
        n += 1
    hub.flush_all()
    summary = {
        "type": "summary",
        "messages": n,
        **store.snapshot(),
        "alerts": store.list_alerts(limit=50),
        "explanations": store.list_explanations(limit=20),
    }
    print(json.dumps(summary, default=str))
    return 0


def _pump_stdin(q: queue.Queue) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            q.put(json.loads(line))
        except Exception as exc:
            log.warning("stdin parse error: %s", exc)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "stdin":
        return cmd_stdin(args)
    parser.error(f"unknown command {args.cmd}")
    return 2
