"""Google Gemini provider (Vertex AI express mode with API key)."""
from __future__ import annotations

import json
import os
from typing import Any

from .base import Narrative
from .parse import parse_narrative

DEFAULT_MODEL = "gemini-3-flash-preview"


class GeminiLLM:
    """Mirrors the TS client: GoogleGenAI({ vertexai: true, apiKey: GEMINI_API_KEY })."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, vertexai: bool | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for the gemini provider")
        if vertexai is None:
            vertexai = os.environ.get("GEMINI_VERTEXAI", "true").lower() not in ("0", "false", "no")
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self._client = genai.Client(vertexai=vertexai, api_key=key)

    def narrate(self, system_prompt: str, packet: dict[str, Any]) -> Narrative:
        from google.genai import types

        user = (
            "Explain this pump alert for an operator. Use ONLY the JSON packet. "
            "Return a single JSON object with keys: summary, likely_cause, "
            "what_to_check (array of strings), do_not_do (array of strings), uncertainty_note. "
            "Do not change fault_type. Do not invent sensors or measurements.\n\n"
            f"{json.dumps(packet, indent=2, default=str)}"
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return parse_narrative(response.text or "", packet, self.name, self.model)
