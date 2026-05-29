"""Environment + path config for the nlp layer (self-contained; does not import frozen spike/)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from schemas import Paper

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LLM (OpenAI-compatible; DeepSeek default) ---
LLM_API_KEY = os.getenv("LLM_API_KEY") or None
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# When true, citation intent dispatches to B's (not-yet-implemented) SciBERT/SciCite model.
USE_SCICITE_MODEL = os.getenv("USE_SCICITE_MODEL", "").lower() in {"1", "true", "yes"}

# --- cache paths (resolved against project root, robust to cwd) ---
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
PAPERS_JSON = CACHE_DIR / "papers.json"
METHOD_CARDS_DIR = CACHE_DIR / "method_cards"


def load_papers() -> list[Paper]:
    """Load the data-layer corpus from data/cache/papers.json (consume only; never write)."""
    if not PAPERS_JSON.exists():
        raise FileNotFoundError(
            f"{PAPERS_JSON} not found. Run the data layer first (see CLAUDE.md)."
        )
    raw = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else list(raw.values())
    return [Paper.from_dict(r) for r in records]
