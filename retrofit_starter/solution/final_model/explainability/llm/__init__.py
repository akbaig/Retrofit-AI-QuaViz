"""Pluggable LLM providers (Gemini now; Ollama later; template fallback)."""
from .base import LLM, Narrative
from .factory import get_llm

__all__ = ["LLM", "Narrative", "get_llm"]
