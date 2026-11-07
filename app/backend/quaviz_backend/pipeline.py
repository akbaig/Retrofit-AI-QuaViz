"""Per-asset ingest → 1-min aggregate → frozen best → optional explain."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import Settings
from .paths import bootstrap_imports
from .store import EventStore

log = logging.getLogger(__name__)


class AssetPipeline:
    def __init__(self, asset_id: str, settings: Settings, store: EventStore, jobs=None):
        bootstrap_imports(settings.challenge_dir)
        from deploy.buffer import MinuteBuffer
        from deploy.runner import OnlineBest

        self.asset_id = asset_id
        self.settings = settings
        self.store = store
        self.jobs = jobs
        self.online = OnlineBest(
            model_path=settings.model_path,
            policy_path=settings.policy_path,
            data_dir=settings.data_dir,
            asset_id=asset_id,
        )
        self.buf = MinuteBuffer(asset_id=asset_id)
        self.minutes = 0
        self.store.ensure_asset(asset_id)

    def ingest(self, payload: dict) -> None:
        ts = payload.get("timestamp_utc")
        self.store.touch_message(self.asset_id, str(ts) if ts is not None else None)
        for minute_frame in self.buf.push(payload):
            self._on_minute(minute_frame)

    def flush(self) -> None:
        for minute_frame in self.buf.flush():
            self._on_minute(minute_frame)
        # Final score on the caller thread (flush / shutdown).
        self.score(enqueue_explain=bool(self.jobs))

    def _on_minute(self, minute_frame) -> None:
        self.online.extend_features(minute_frame)
        self.minutes += 1
        self.store.set_progress(
            self.asset_id,
            minutes=self.minutes,
            feature_rows=int(len(self.online.features)),
        )
        if self.settings.infer_every > 0 and self.minutes % self.settings.infer_every == 0:
            # Keep MQTT ingest hot-path free of model + LLM work.
            if self.jobs is not None:
                self.jobs.request_infer(self.asset_id)
            else:
                self.score(enqueue_explain=False)

    def score(self, *, enqueue_explain: bool = False) -> list[dict[str, Any]]:
        fresh = self.online.infer_new_alerts()
        self.store.set_progress(
            self.asset_id,
            minutes=self.minutes,
            feature_rows=int(len(self.online.features)),
            last_infer_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        if not len(fresh):
            return []
        out = []
        for _, row in fresh.iterrows():
            rec = row.to_dict()
            stored = self.store.add_alert(rec)
            out.append(stored)
            log.info(
                "alert %s %s conf=%.3f",
                rec.get("asset_id"),
                rec.get("fault_type"),
                float(rec.get("confidence") or 0),
            )
            if self.settings.explain:
                if enqueue_explain and self.jobs is not None:
                    self.jobs.request_explain(self, stored)
                else:
                    self._explain(stored)
        return out

    def _explain(self, alert: dict[str, Any]) -> None:
        try:
            card = explain_live_alert(
                online=self.online,
                alert=alert,
                settings=self.settings,
            )
            self.store.add_explanation(
                {
                    "asset_id": alert["asset_id"],
                    "alert_id": alert.get("id"),
                    "alert_time_utc": alert["alert_time_utc"],
                    "fault_type": alert["fault_type"],
                    "narrative": card.get("narrative"),
                    "packet": card.get("packet"),
                    "sandbox_action": (card.get("packet") or {}).get("sandbox_action"),
                    "what_if": card.get("what_if"),
                    "provider": card.get("provider"),
                }
            )
        except Exception as exc:
            log.exception("explain failed for %s", alert.get("fault_type"))
            self.store.add_error(
                {
                    "asset_id": alert.get("asset_id"),
                    "alert_time_utc": alert.get("alert_time_utc"),
                    "error": str(exc),
                }
            )


class PipelineHub:
    """Multi-asset registry; routes MQTT / stdin payloads to AssetPipeline."""

    def __init__(self, settings: Settings, store: EventStore | None = None, jobs=None):
        self.settings = settings
        self.store = store or EventStore()
        self.jobs = jobs
        self._pipelines: dict[str, AssetPipeline] = {}
        self._known: set[str] | None = None
        bootstrap_imports(settings.challenge_dir)
        try:
            assets = pd.read_csv(settings.data_dir / "assets.csv")
            self._known = set(assets.asset_id.astype(str))
        except Exception:
            self._known = None
        if settings.asset_ids:
            for aid in settings.asset_ids:
                self._pipelines[aid] = AssetPipeline(aid, settings, self.store, jobs=self.jobs)

    def attach_jobs(self, jobs) -> None:
        self.jobs = jobs
        for pipe in self._pipelines.values():
            pipe.jobs = jobs

    def allowed(self, asset_id: str) -> bool:
        if self.settings.asset_ids and asset_id not in self.settings.asset_ids:
            return False
        if self._known is not None and asset_id not in self._known:
            return False
        return True

    def get(self, asset_id: str) -> AssetPipeline | None:
        if not self.allowed(asset_id):
            return None
        pipe = self._pipelines.get(asset_id)
        if pipe is None:
            pipe = AssetPipeline(asset_id, self.settings, self.store, jobs=self.jobs)
            self._pipelines[asset_id] = pipe
        return pipe

    def ingest_message(self, msg: dict[str, Any]) -> None:
        from deploy.aggregate import asset_id_from_topic

        if "payload" in msg and isinstance(msg["payload"], dict):
            topic = msg.get("topic")
            payload = dict(msg["payload"])
            aid = asset_id_from_topic(topic) or payload.get("asset_id")
        else:
            payload = dict(msg)
            aid = payload.get("asset_id")
        if not aid:
            raise ValueError("asset_id missing from topic/payload")
        aid = str(aid)
        pipe = self.get(aid)
        if pipe is None:
            return
        payload["asset_id"] = aid
        pipe.ingest(payload)

    def flush_all(self) -> None:
        for pipe in list(self._pipelines.values()):
            pipe.flush()


def explain_live_alert(*, online, alert: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Build explain card from the live aggregated feature table (not parquet)."""
    bootstrap_imports(settings.challenge_dir)
    from explainability.llm.factory import get_llm
    from explainability.narrate import narrate_alert
    from explainability.packet import explain_alert
    from explainability.whatif_eval import evaluate_what_if

    llm = get_llm(settings.llm_provider)
    provider = type(llm).__name__

    # Re-run explain_alert on the growing online feature frame, then match this alert.
    target_t = pd.Timestamp(alert["alert_time_utc"])
    target_f = str(alert["fault_type"])
    # First pass discovers alert list; second pass (if needed) selects the matching card.
    explained = explain_alert(
        model=online.model,
        policy=online.policy,
        features=online.features,
        asset=online.asset,
        context=online.context,
        snapshots=online.snapshots,
        alert_index=0,
    )
    match_idx = 0
    alerts = explained["alerts"]
    for i in range(len(alerts)):
        row = alerts.iloc[i]
        if str(row.fault_type) != target_f:
            continue
        if abs((pd.Timestamp(row.alert_time_utc) - target_t).total_seconds()) < 1.0:
            match_idx = i
            break
    if match_idx != 0:
        explained = explain_alert(
            model=online.model,
            policy=online.policy,
            features=online.features,
            asset=online.asset,
            context=online.context,
            snapshots=online.snapshots,
            alert_index=match_idx,
        )

    packet = explained["packet"]
    narrative = narrate_alert(packet, llm=llm)
    out: dict[str, Any] = {
        "alert": explained["alert"],
        "packet": packet,
        "narrative": narrative.to_dict(),
        "provider": provider,
    }
    if settings.what_if:
        out["what_if"] = evaluate_what_if(
            packet,
            settings.data_dir,
            challenge_root=settings.challenge_dir,
        )
    return out
