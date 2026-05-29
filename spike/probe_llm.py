"""Standalone probe: extract ONE MethodCard from one paper via the LLM API for manual inspection."""

from __future__ import annotations

import json
import sys

from schemas import MethodCard, Paper

from . import config
from .fetch import load_papers

_PROMPT = """You are an expert at reading machine-learning papers. Extract a \
mechanism-level "method card" from the paper below. Return ONLY a JSON object with \
these keys: name, problem, method, key_idea, datasets (list), metrics (list), \
baselines (list). Be concise and factual; use null or [] when unknown.

Title: {title}

Abstract:
{abstract}
"""


def extract_method_card(paper: Paper) -> MethodCard:
    """Call the LLM to fill a MethodCard skeleton for one paper."""
    from openai import OpenAI

    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY not set. Add it to .env (DeepSeek/OpenAI-compatible).")
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": _PROMPT.format(title=paper.title, abstract=paper.abstract),
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    data["paper_id"] = paper.paper_id
    return MethodCard.from_dict(data)


def main() -> None:
    """Pick the most-cited cached paper, extract a method card, and pretty-print it."""
    papers = load_papers()
    if not papers:
        print("No papers cached. Run `uv run python -m spike` first.", file=sys.stderr)
        sys.exit(1)
    paper = max(papers, key=lambda p: p.citation_count)
    print(f"[probe] paper: {paper.title} ({paper.year}, cites={paper.citation_count})\n")
    try:
        card = extract_method_card(paper)
    except RuntimeError as e:
        print(f"[probe] skipped: {e}", file=sys.stderr)
        return
    print(json.dumps(card.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
