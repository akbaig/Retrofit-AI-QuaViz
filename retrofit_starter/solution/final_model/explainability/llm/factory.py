"""Select an LLM provider from env (swap gemini ↔ ollama without code changes)."""
from __future__ import annotations

import os
from pathlib import Path

from .base import LLM
from .gemini import GeminiLLM
from .ollama import OllamaLLM
from .template import TemplateLLM


def load_dotenv(path: Path | None = None) -> None:
    """Load repo-root .env if python-dotenv is installed; no-op otherwise."""
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    if path is not None:
        _load(path, override=False)
        return
    here = Path(__file__).resolve()
    for candidate in (here.parents[5] / ".env", here.parents[4] / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            _load(candidate, override=False)
            return


def get_llm(provider: str | None = None) -> LLM:
    """
    provider: gemini | ollama | template (default from LLM_PROVIDER, else gemini).
    Falls back to template if the chosen provider cannot start.
    """
    load_dotenv()
    name = (provider or os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    if name in ("template", "none", "offline"):
        return TemplateLLM()
    if name == "ollama":
        return OllamaLLM()
    if name in ("gemini", "google", "google-genai"):
        try:
            return GeminiLLM()
        except Exception:
            if os.environ.get("LLM_REQUIRE_PROVIDER", "").lower() in ("1", "true", "yes"):
                raise
            return TemplateLLM()
    raise ValueError(f"Unknown LLM_PROVIDER={name!r}; use gemini, ollama, or template")
