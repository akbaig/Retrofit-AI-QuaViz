"""Background infer / explain workers so MQTT ingest never blocks on Gemini."""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any

log = logging.getLogger(__name__)


class BackgroundJobs:
    def __init__(self, hub, *, explain_max: int = 64):
        self.hub = hub
        self.infer_q: queue.Queue = queue.Queue()
        self.explain_q: queue.Queue = queue.Queue(maxsize=explain_max)
        self._infer_pending: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for name, target in (
            ("infer-worker", self._infer_loop),
            ("explain-worker", self._explain_loop),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self.infer_q.put(None)
        self.explain_q.put(None)
        for t in self._threads:
            t.join(timeout=60)

    def request_infer(self, asset_id: str) -> None:
        with self._lock:
            if asset_id in self._infer_pending:
                return
            self._infer_pending.add(asset_id)
        self.infer_q.put(asset_id)

    def request_explain(self, pipe, alert: dict[str, Any]) -> None:
        try:
            self.explain_q.put_nowait((pipe, alert))
        except queue.Full:
            log.warning(
                "explain backlog full; skipping %s %s",
                alert.get("asset_id"),
                alert.get("fault_type"),
            )

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                asset_id = self.infer_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if asset_id is None:
                break
            try:
                pipe = self.hub._pipelines.get(asset_id)
                if pipe is not None:
                    pipe.score(enqueue_explain=True)
            except Exception:
                log.exception("infer failed for %s", asset_id)
            finally:
                with self._lock:
                    self._infer_pending.discard(asset_id)

    def _explain_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.explain_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            pipe, alert = item
            try:
                pipe._explain(alert)
            except Exception:
                log.exception("explain worker failed")
