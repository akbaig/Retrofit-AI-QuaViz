"""HTTP API for the operator frontend."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import PipelineHub
from .store import EventStore


def create_app(hub: PipelineHub, store: EventStore | None = None) -> FastAPI:
    store = store or hub.store
    app = FastAPI(title="QuaViz backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.hub = hub
    app.state.store = store

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "quaviz-backend", "docs": "/docs", "health": "/health"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        s = hub.settings
        return {
            "ok": True,
            "model": str(s.model_path),
            "data": str(s.data_dir),
            "explain": s.explain,
            "infer_every": s.infer_every,
            "mqtt_host": s.mqtt_host,
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        return store.snapshot()

    @app.get("/assets")
    def assets() -> dict[str, Any]:
        snap = store.snapshot()
        return {"assets": list(snap["assets"].values())}

    @app.get("/alerts")
    def alerts(
        asset_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        return {"alerts": store.list_alerts(asset_id=asset_id, limit=limit)}

    @app.get("/explanations")
    def explanations(
        asset_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        return {"explanations": store.list_explanations(asset_id=asset_id, limit=limit)}

    @app.post("/score/{asset_id}")
    def score_asset(asset_id: str) -> dict[str, Any]:
        pipe = hub.get(asset_id)
        if pipe is None:
            raise HTTPException(404, f"asset {asset_id} not allowed or unknown")
        fresh = pipe.score()
        return {"alerts": fresh}

    @app.post("/flush")
    def flush() -> dict[str, Any]:
        hub.flush_all()
        return store.snapshot()

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        import queue as qmod

        await ws.accept()
        q = store.subscribe()

        def _next():
            try:
                return q.get(timeout=1.0)
            except qmod.Empty:
                return None

        try:
            while True:
                event = await asyncio.get_event_loop().run_in_executor(None, _next)
                if event is None:
                    continue
                await ws.send_text(json.dumps(event, default=str))
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(q)

    return app
