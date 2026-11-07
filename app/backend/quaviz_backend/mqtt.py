"""MQTT subscriber: retrofit/<site>/<asset>/telemetry -> ingest queue."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)


class MqttSubscriber:
    """Background paho client; pushes {topic, payload} dicts onto a queue."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 1883,
        topic: str = "retrofit/+/+/telemetry",
        client_id: str = "quaviz-backend",
        out: queue.Queue | None = None,
    ):
        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.out = out if out is not None else queue.Queue(maxsize=100_000)
        self._client = None
        self._connected = threading.Event()
        self._drop_count = 0
        self._drop_log_at = 0.0

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "paho-mqtt is required for MQTT mode; pip install paho-mqtt"
            ) from exc

        try:
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
            version2 = True
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=self.client_id)
            version2 = False

        def on_connect(client, userdata, flags, reason_code, properties=None):
            rc = reason_code
            if hasattr(reason_code, "value"):
                rc = reason_code.value
            if int(rc) != 0:
                log.error("MQTT connect failed rc=%s", reason_code)
                return
            client.subscribe(self.topic, qos=0)
            self._connected.set()
            log.info("MQTT subscribed %s @ %s:%s", self.topic, self.host, self.port)

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception as exc:
                log.warning("bad MQTT payload on %s: %s", msg.topic, exc)
                return
            try:
                self.out.put_nowait({"topic": msg.topic, "payload": payload})
            except queue.Full:
                self._drop_count += 1
                now = time.monotonic()
                if now - self._drop_log_at >= 2.0:
                    log.warning(
                        "ingest queue full; dropped %s msgs (latest %s)",
                        self._drop_count,
                        msg.topic,
                    )
                    self._drop_count = 0
                    self._drop_log_at = now

        def on_disconnect(client, userdata, *args):
            self._connected.clear()
            log.warning(
                "MQTT disconnected from %s:%s - will retry (is a broker running?)",
                self.host,
                self.port,
            )

        if version2:
            client.on_connect = on_connect
        else:
            def on_connect_v1(client, userdata, flags, rc):
                on_connect(client, userdata, flags, rc, None)

            client.on_connect = on_connect_v1
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.connect_async(self.host, self.port, keepalive=60)
        client.loop_start()
        self._client = client
        log.info(
            "MQTT connecting to %s:%s (topic %s) - waiting for broker if needed",
            self.host,
            self.port,
            self.topic,
        )

    def stop(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
        self._client = None
        self._connected.clear()


class IngestWorker:
    """Drain queue -> hub.ingest_message; runs in a daemon thread."""

    def __init__(self, hub, q: queue.Queue, on_error: Callable | None = None):
        self.hub = hub
        self.q = q
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)
        self._thread.start()

    def stop(self, flush: bool = True) -> None:
        self._stop.set()
        self.q.put(None)
        if self._thread:
            self._thread.join(timeout=30)
        if flush:
            self.hub.flush_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                if self._stop.is_set():
                    break
                continue
            try:
                self.hub.ingest_message(item)
            except Exception as exc:
                log.exception("ingest failed")
                if self.on_error:
                    self.on_error({"error": str(exc), "item_keys": list(item)})
