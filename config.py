"""
TARKA — Configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ── LLM Provider (Shared) ─────────────────────────────────────
# "ollama" (local) or "gemini" (cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ── Ollama / Local LLM ────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1800"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
OLLAMA_THINK_FOR_THEORY = True

# ── Gemini / Cloud LLM ────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "120"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# ── API Server ────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8901

# ── Reasoning Thresholds ──────────────────────────────────────
TEMPORAL_WINDOW_HOURS = 24          # events within this window are correlated
MIN_CONFIDENCE_THRESHOLD = 0.3      # edges below this are flagged for gap-bridging
GAP_BRIDGE_LLM_CONFIDENCE = 0.5     # confidence for LLM-inferred edges
GAP_BRIDGE_SEARCH_CONFIDENCE = 0.7  # confidence for search-verified edges
FACT_CONFIDENCE = 1.0               # explicit confirmed facts
ENABLE_SEMANTIC_GAP_DETECTION = True # use LLM to detect logical gaps (adds ~30s)

# ── GDELT ─────────────────────────────────────────────────────
GDELT_API_URL = "http://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DEFAULT_TIMESPAN = 60        # minutes to look back

# ── DuckDuckGo ────────────────────────────────────────────────
DDG_MAX_RESULTS = 5

# ── Product ───────────────────────────────────────────────────
PRODUCT_NAME = "TARKA"
PRODUCT_VERSION = "0.1.0"
PRODUCT_TAGLINE = "Agentic Graph-RAG OSINT Intelligence"
