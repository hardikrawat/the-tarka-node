"""
TARKA — Unified LLM Client

Dispatches LLM calls to either Ollama (local) or Gemini (cloud)
based on config.LLM_PROVIDER.

All agents import from here — never directly from ollama_client
or gemini_client.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import config

log = logging.getLogger("tarka.llm")

# Same callback type used everywhere
ProgressCallback = Callable[[int, float, str], None]


async def call_llm(
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    think: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """
    Call the active LLM provider with streaming support.

    Routes to Ollama or Gemini based on config.LLM_PROVIDER.
    Same interface as call_ollama() — drop-in replacement.
    """
    provider = getattr(config, "LLM_PROVIDER", "ollama")

    if provider == "gemini":
        api_key = getattr(config, "GEMINI_API_KEY", "")
        if not api_key:
            log.warning("GEMINI_API_KEY not set, falling back to Ollama")
            provider = "ollama"

    if provider == "gemini":
        from agents.gemini_client import call_gemini
        return await call_gemini(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            on_progress=on_progress,
        )
    else:
        from agents.ollama_client import call_ollama
        return await call_ollama(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            on_progress=on_progress,
        )
