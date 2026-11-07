"""Ollama local provider (OpenAI-compatible /api/chat)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import Narrative
from .parse import parse_narrative

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE = "http://127.0.0.1:11434"


class OllamaLLM:
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout_s: float = 120.0):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s

    def narrate(self, system_prompt: str, packet: dict[str, Any]) -> Narrative:
        user = (
            "Explain this pump alert for an operator. Use ONLY the JSON packet. "
            "Return a single JSON object with keys: summary, likely_cause, "
            "what_to_check (array of strings), do_not_do (array of strings), uncertainty_note. "
            "Do not change fault_type. Do not invent sensors or measurements.\n\n"
            f"{json.dumps(packet, indent=2, default=str)}"
        )
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url}. Start Ollama or set LLM_PROVIDER=gemini|template."
            ) from exc
        text = (payload.get("message") or {}).get("content") or ""
        return parse_narrative(text, packet, self.name, self.model)
