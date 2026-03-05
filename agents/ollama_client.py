"""
TARKA — Shared Ollama Client

Centralised Ollama API client with:
- Streaming support for real-time progress tracking
- Retry logic with exponential backoff
- Configurable thinking mode
- Progress callbacks for TUI integration
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import httpx

import config

log = logging.getLogger("tarka.ollama")

# Type alias for progress callbacks
# callback(token_count: int, elapsed_seconds: float, partial_text: str)
ProgressCallback = Callable[[int, float, str], None]


async def call_ollama(
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    think: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """
    Call Ollama chat API with streaming for progress updates and
    retry logic for resilience on local hardware.

    Args:
        prompt:      The user prompt to send
        temperature: LLM temperature (lower = more deterministic)
        max_tokens:  Maximum tokens to generate
        think:       Enable thinking mode (useful for complex reasoning)
        on_progress: Optional callback for real-time progress updates

    Returns:
        The complete assistant message content.
    """
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # always stream for progress tracking
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    timeout = httpx.Timeout(
        connect=30.0,
        read=config.OLLAMA_TIMEOUT,
        write=30.0,
        pool=30.0,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, config.OLLAMA_MAX_RETRIES + 1):
        try:
            content = await _stream_ollama(url, payload, timeout, on_progress)
            if content:
                return content
            log.warning(f"Ollama returned empty content (attempt {attempt})")
        except httpx.TimeoutException as exc:
            log.warning(
                f"Ollama request timed out "
                f"(attempt {attempt}/{config.OLLAMA_MAX_RETRIES})"
            )
            last_exc = exc
        except Exception as exc:
            log.error(f"Ollama request failed: {exc}")
            last_exc = exc

        if attempt < config.OLLAMA_MAX_RETRIES:
            backoff = 2 ** attempt
            log.info(f"Retrying in {backoff}s...")
            await asyncio.sleep(backoff)

    log.error(
        f"Ollama failed after {config.OLLAMA_MAX_RETRIES} attempts: {last_exc}"
    )
    return ""


async def _stream_ollama(
    url: str,
    payload: dict,
    timeout: httpx.Timeout,
    on_progress: Optional[ProgressCallback],
) -> str:
    """Stream the Ollama response, firing progress callbacks per chunk.
    
    Handles long silent "thinking" phases by sending heartbeat
    progress updates so the TUI stays alive.
    """
    collected: list[str] = []
    token_count = 0
    start_time = time.monotonic()
    last_token_time = start_time

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 500:
                body = await resp.aread()
                log.error(
                    f"Ollama returned HTTP {resp.status_code}: {body[:200]}"
                )
                raise httpx.HTTPStatusError(
                    f"Ollama server error {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    import json
                    chunk = json.loads(line)
                except Exception:
                    continue

                now = time.monotonic()

                # Extract content token
                token = chunk.get("message", {}).get("content", "")
                if token:
                    collected.append(token)
                    token_count += 1
                    last_token_time = now

                    # Fire progress callback (non-blocking)
                    if on_progress:
                        elapsed = now - start_time
                        try:
                            on_progress(token_count, elapsed, token)
                        except Exception as exc:
                            log.error(f"Progress callback failed: {exc}")
                else:
                    # No content token — LLM is still "thinking"
                    # Send heartbeat to keep UI alive
                    if on_progress and (now - last_token_time) > 5.0:
                        elapsed = now - start_time
                        thinking_secs = int(now - last_token_time)
                        try:
                            on_progress(
                                token_count,
                                elapsed,
                                "",  # empty token = heartbeat
                            )
                        except Exception:
                            pass

                # Check if done
                if chunk.get("done", False):
                    break

    return "".join(collected)

