"""Backend settings from environment / CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    challenge_dir,
    data_dir,
    model_path,
    policy_path,
    repo_dotenv,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env = repo_dotenv()
    if env.is_file():
        load_dotenv(env, override=False)
    load_dotenv(override=False)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class Settings:
    challenge_dir: Path
    data_dir: Path
    model_path: Path
    policy_path: Path

    mqtt_host: str | None = None
    mqtt_port: int = 1883
    mqtt_topic: str = "retrofit/+/+/telemetry"
    mqtt_client_id: str = "quaviz-backend"

    http_host: str = "0.0.0.0"
    http_port: int = 8000

    asset_ids: list[str] = field(default_factory=list)  # empty = accept any known / any topic
    infer_every: int = 5  # 0 = only on flush / explicit score
    explain: bool = True
    llm_provider: str | None = None  # gemini | ollama | template
    what_if: bool = False

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        _load_dotenv()
        assets_raw = os.environ.get("QUAVIZ_ASSETS", "").strip()
        assets = [a.strip() for a in assets_raw.split(",") if a.strip()]
        s = cls(
            challenge_dir=challenge_dir(),
            data_dir=data_dir(),
            model_path=model_path(),
            policy_path=policy_path(),
            mqtt_host=os.environ.get("MQTT_HOST") or None,
            mqtt_port=_int("MQTT_PORT", 1883),
            mqtt_topic=os.environ.get("MQTT_TOPIC", "retrofit/+/+/telemetry"),
            mqtt_client_id=os.environ.get("MQTT_CLIENT_ID", "quaviz-backend"),
            http_host=os.environ.get("HTTP_HOST", "0.0.0.0"),
            http_port=_int("HTTP_PORT", 8000),
            asset_ids=assets,
            infer_every=_int("INFER_EVERY", 5),
            explain=_bool("EXPLAIN", True),
            llm_provider=os.environ.get("LLM_PROVIDER") or None,
            what_if=_bool("EXPLAIN_WHAT_IF", False),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(s, k):
                setattr(s, k, v)
        return s
