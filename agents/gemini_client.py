"""
TARKA — Gemini Cloud Client

Google Gemini API client with:
- Async streaming for real-time progress tracking
- Retry logic with exponential backoff
- Progress callbacks for TUI integration
- Same interface as ollama_client.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import config

log = logging.getLogger("tarka.gemini")

# Type alias (same as ollama_client.py)
ProgressCallback = Callable[[int, float, str], None]


async def call_gemini(
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    think: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """
    Call Google Gemini API with streaming and retry logic.

    Uses the google-genai SDK (pip install google-genai).
    API key is read from config.GEMINI_API_KEY.
    """
    api_key = getattr(config, "GEMINI_API_KEY", "")
    if not api_key:
        log.error("GEMINI_API_KEY is not set")
        return ""

    model = getattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
    max_retries = getattr(config, "GEMINI_MAX_RETRIES", 3)
    timeout = getattr(config, "GEMINI_TIMEOUT", 120)

    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            content = await _stream_gemini(
                api_key=api_key,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                think=think,
                timeout=timeout,
                on_progress=on_progress,
            )
            if content:
                return content
            log.warning(f"Gemini returned empty content (attempt {attempt})")
        except asyncio.TimeoutError as exc:
            log.warning(
                f"Gemini request timed out "
                f"(attempt {attempt}/{max_retries})"
            )
            last_exc = exc
        except Exception as exc:
            import re
            
            error_str = str(exc)
            delay = 0

            # Handle 429 Rate Limits specifically
            if "429" in error_str or "Too Many Requests" in error_str:
                log.warning("Gemini Rate Limit Exceeded (429)")
                # Try to extract the retry delay from the error message
                match = re.search(r"retryDelay(?:[\"']\s*:\s*[\"']?|.*?\s+)((?:\d+\.)?\d+)s", error_str)
                if not match:
                    match = re.search(r"Please retry in ((?:\d+\.)?\d+)s", error_str)
                
                if match:
                    try:
                        delay = float(match.group(1)) + 1.0  # Add 1s buffer
                    except ValueError:
                        pass
                
                if delay <= 0:
                    delay = (2 ** attempt) * 5  # Harsher default backoff for 429: 10s, 20s, 40s
            else:
                log.error(f"Gemini request failed: {exc}")
                delay = 2 ** attempt

            last_exc = exc

        if attempt < max_retries:
            log.info(f"Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    log.error(f"Gemini failed after {max_retries} attempts: {last_exc}")
    return ""


async def _stream_gemini(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    think: bool,
    timeout: float,
    on_progress: Optional[ProgressCallback],
) -> str:
    """Stream a Gemini response, firing progress callbacks per chunk."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log.error(
            "google-genai package not installed. "
            "Run: pip install google-genai"
        )
        return ""

    import httpx
    
    # The SDK v1 defaults to aiohttp. To enforce a generous read timeout 
    # without keyword collisions, we must bypass aiohttp entirely by explicitly 
    # injecting an httpx AsyncClient into the SDK's options.
    custom_httpx = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    
    client = genai.Client(
        api_key=api_key,
        http_options={"httpx_async_client": custom_httpx}
    )

    # Build config
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )

    # Enable thinking if supported (Gemini 2.5+)
    if think:
        gen_config.thinking_config = types.ThinkingConfig(
            thinking_budget=4096,
        )

    collected: list[str] = []
    token_count = 0
    start_time = time.monotonic()

    try:
        response = await client.aio.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=gen_config,
        )

        async for chunk in response:
            now = time.monotonic()

            # Extract text from chunk
            text = ""
            if chunk.text:
                text = chunk.text

            if text:
                collected.append(text)
                token_count += 1

                if on_progress:
                    elapsed = now - start_time
                    try:
                        on_progress(token_count, elapsed, text)
                    except Exception as exc:
                        log.error(f"Progress callback failed: {exc}")
            else:
                # Thinking/processing heartbeat
                if on_progress:
                    elapsed = now - start_time
                    try:
                        on_progress(token_count, elapsed, "")
                    except Exception:
                        pass

    except Exception as exc:
        log.error(f"Gemini streaming error: {exc}")
        raise

    return "".join(collected)
