"""
TARKA — Entry Point

Launches the TUI dashboard with the FastAPI server running
in a background thread for external OSINT tool integration.

Usage:
    python main.py              # TUI mode (default)
    python main.py --api-only   # Headless API server only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading

import uvicorn

import config
from api.routes import create_api
from graph.knowledge_graph import IntelGraph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TARKA — Agentic Graph-RAG OSINT Intelligence"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run headless API server only (no TUI)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help=f"API server port (default: {config.API_PORT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. qwen3:14b or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "gemini"],
        default=None,
        help="LLM Provider (ollama or gemini)",
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Google Gemini API Key (if using --provider gemini)",
    )
    args = parser.parse_args()

    # Override config if CLI args provided
    if args.provider:
        config.LLM_PROVIDER = args.provider
    if args.gemini_key:
        config.GEMINI_API_KEY = args.gemini_key
    if args.model:
        if config.LLM_PROVIDER == "gemini":
            config.GEMINI_MODEL = args.model
        else:
            config.OLLAMA_MODEL = args.model

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    log = logging.getLogger("tarka")
    
    # Log active configuration
    if config.LLM_PROVIDER == "gemini":
        log.info(f"Using Provider: GEMINI CLOUD (Model: {config.GEMINI_MODEL})")
        if not config.GEMINI_API_KEY:
            log.warning("GEMINI_API_KEY is not set! API calls will fail.")
    else:
        log.info(f"Using Provider: OLLAMA LOCAL (Model: {config.OLLAMA_MODEL})")

    # Create shared graph instance
    graph = IntelGraph()

    if args.api_only:
        # ── Headless API mode ─────────────────────────────────
        log.info(
            f"TARKA API server starting on http://{config.API_HOST}:{args.port}"
        )
        api = create_api(graph)
        uvicorn.run(api, host=config.API_HOST, port=args.port, log_level="info")
    else:
        # ── TUI + Background API mode ────────────────────────
        api = create_api(graph)

        # Start API server in background thread
        api_thread = threading.Thread(
            target=_run_api_server,
            args=(api, args.port),
            daemon=True,
        )
        api_thread.start()
        log.info(f"API server started on http://localhost:{args.port}")

        # Remove stderr handler — TUI has its own log panel
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                root_logger.removeHandler(handler)
        tarka_logger = logging.getLogger("tarka")
        for handler in tarka_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                tarka_logger.removeHandler(handler)

        # Launch TUI
        from tui.app import TarkaApp
        app = TarkaApp(graph=graph)
        app.run()


def _run_api_server(api, port: int) -> None:
    """Run uvicorn in a background thread."""
    uv_config = uvicorn.Config(
        app=api,
        host=config.API_HOST,
        port=port,
        log_level="warning",  # quiet in TUI mode
    )
    server = uvicorn.Server(uv_config)
    server.run()


if __name__ == "__main__":
    main()
