"""Thread-safe in-memory event store for API / future WebSocket."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssetStatus:
    asset_id: str
    messages: int = 0
    minutes_aggregated: int = 0
    feature_rows: int = 0
    alerts: int = 0
    last_timestamp_utc: str | None = None
    last_infer_utc: str | None = None


@dataclass
class EventStore:
    max_alerts: int = 500
    max_explanations: int = 200
    max_events: int = 1000
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    started_at: float = field(default_factory=time.time)
    assets: dict[str, AssetStatus] = field(default_factory=dict)
    alerts: deque = field(default_factory=deque)
    explanations: deque = field(default_factory=deque)
    events: deque = field(default_factory=deque)
    _subscribers: list = field(default_factory=list, repr=False)

    def ensure_asset(self, asset_id: str) -> AssetStatus:
        with self._lock:
            if asset_id not in self.assets:
                self.assets[asset_id] = AssetStatus(asset_id=asset_id)
            return self.assets[asset_id]

    def touch_message(self, asset_id: str, timestamp_utc: str | None) -> None:
        with self._lock:
            st = self.assets.setdefault(asset_id, AssetStatus(asset_id=asset_id))
            st.messages += 1
            if timestamp_utc:
                st.last_timestamp_utc = str(timestamp_utc)

    def set_progress(
        self,
        asset_id: str,
        *,
        minutes: int | None = None,
        feature_rows: int | None = None,
        last_infer_utc: str | None = None,
    ) -> None:
        with self._lock:
            st = self.assets.setdefault(asset_id, AssetStatus(asset_id=asset_id))
            if minutes is not None:
                st.minutes_aggregated = minutes
            if feature_rows is not None:
                st.feature_rows = feature_rows
            if last_infer_utc is not None:
                st.last_infer_utc = last_infer_utc

    def add_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        rec = dict(alert)
        rec.setdefault("id", str(uuid.uuid4()))
        rec.setdefault("received_at", time.time())
        with self._lock:
            self.alerts.appendleft(rec)
            while len(self.alerts) > self.max_alerts:
                self.alerts.pop()
            st = self.assets.setdefault(rec["asset_id"], AssetStatus(asset_id=rec["asset_id"]))
            st.alerts += 1
            event = {"type": "alert", **rec}
            self._push_event(event)
        return rec

    def add_explanation(self, explanation: dict[str, Any]) -> dict[str, Any]:
        rec = dict(explanation)
        rec.setdefault("id", str(uuid.uuid4()))
        rec.setdefault("received_at", time.time())
        with self._lock:
            self.explanations.appendleft(rec)
            while len(self.explanations) > self.max_explanations:
                self.explanations.pop()
            event = {"type": "explain", **rec}
            self._push_event(event)
        return rec

    def add_error(self, error: dict[str, Any]) -> None:
        with self._lock:
            self._push_event({"type": "error", **error, "received_at": time.time()})

    def _push_event(self, event: dict[str, Any]) -> None:
        self.events.appendleft(event)
        while len(self.events) > self.max_events:
            self.events.pop()
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def subscribe(self):
        import queue

        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "assets": {k: vars(v).copy() for k, v in self.assets.items()},
                "alert_count": len(self.alerts),
                "explanation_count": len(self.explanations),
            }

    def list_alerts(self, asset_id: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = list(self.alerts)
        if asset_id:
            rows = [r for r in rows if r.get("asset_id") == asset_id]
        return rows[:limit]

    def list_explanations(self, asset_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = list(self.explanations)
        if asset_id:
            rows = [r for r in rows if r.get("asset_id") == asset_id]
        return rows[:limit]
