"""LLM-backed method card extraction: build prompt, call LLM, robust-parse JSON, cache.

CLI:
    uv run python -m nlp.method_card.extractor --top 50 [--force] [--dry-run]
    uv run python -m nlp.method_card.extractor --sample 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from schemas import MethodCard, Paper

from .. import config
from .prompts import build_prompt

# Fields reported in the per-field non-empty rate (the extractor's quality signal).
_REPORT_FIELDS = ["task", "input", "output", "backbone", "loss", "key_idea", "datasets", "metrics"]


def _parse_json(raw: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating code fences and prose wrapping.

    Raises json.JSONDecodeError if no JSON object can be recovered.
    """
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _card_path(paper_id: str):
    """Cache path for one paper's method card (paper_id is an OpenAlex/S2 id, filename-safe)."""
    safe = paper_id.replace("/", "_")
    return config.METHOD_CARDS_DIR / f"{safe}.json"


def _estimate_input_tokens(messages: list[dict]) -> int:
    """Rough input-token estimate (~4 chars/token) for one prompt."""
    return sum(len(m["content"]) for m in messages) // 4


class MethodCardExtractor:
    """Extract MethodCards from papers via an OpenAI-compatible LLM, caching per paper."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None  # lazily constructed so tests can inject a mock

    @property
    def client(self):
        """Lazily build the OpenAI-compatible client (deferred so tests need no real key)."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call_llm(self, messages: list[dict]) -> str:
        """Single chat completion; returns raw message content."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def extract_one(self, paper: Paper) -> MethodCard | None:
        """Extract one method card; one retry on bad JSON; None on persistent/API failure."""
        messages = build_prompt(paper.abstract or "", paper.title)
        for attempt in range(2):
            try:
                raw = self._call_llm(messages)
            except Exception as e:  # network / API / auth error: do not retry
                print(f"  [warn] LLM call failed for {paper.paper_id}: {e}", file=sys.stderr)
                return None
            try:
                data = _parse_json(raw)
            except json.JSONDecodeError:
                if attempt == 0:
                    continue  # retry once
                print(f"  [warn] unparseable JSON for {paper.paper_id}", file=sys.stderr)
                return None
            data["paper_id"] = paper.paper_id
            return MethodCard.from_dict(data)
        return None

    def extract_batch(self, papers: list[Paper], force: bool = False) -> list[MethodCard]:
        """Extract cards for papers, caching each at data/cache/method_cards/{paper_id}.json."""
        config.METHOD_CARDS_DIR.mkdir(parents=True, exist_ok=True)
        cards: list[MethodCard] = []
        for i, paper in enumerate(papers, 1):
            path = _card_path(paper.paper_id)
            if path.exists() and not force:
                cards.append(MethodCard.from_dict(json.loads(path.read_text(encoding="utf-8"))))
                print(f"  [{i}/{len(papers)}] cached  {paper.paper_id}")
                continue
            print(f"  [{i}/{len(papers)}] extract {paper.paper_id}  {paper.title[:60]}")
            card = self.extract_one(paper)
            if card is None:
                continue
            path.write_text(
                json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cards.append(card)
        return cards


def _is_filled(value) -> bool:
    """True if a field carries content (non-empty string / non-empty list)."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return value is not None


def _print_field_rates(cards: list[MethodCard]) -> None:
    """Print per-field non-empty rate across extracted cards."""
    if not cards:
        print("No cards extracted; nothing to report.")
        return
    parts = []
    for f in _REPORT_FIELDS:
        n = sum(1 for c in cards if _is_filled(getattr(c, f, None)))
        parts.append(f"{f}: {round(100 * n / len(cards))}%")
    print(f"\nPer-field non-empty rate over {len(cards)} cards:\n  " + ", ".join(parts))


def _print_cards(cards: list[MethodCard], n: int) -> None:
    """Pretty-print up to n cards for a human eyeball check."""
    if not cards:
        print("No cached cards found. Run an extraction first.")
        return
    sample = random.sample(cards, min(n, len(cards)))
    for c in sample:
        print("\n" + "=" * 72)
        print(json.dumps(c.to_dict(), ensure_ascii=False, indent=2))


def _load_cached_cards() -> list[MethodCard]:
    """Read all method cards already in the cache directory."""
    if not config.METHOD_CARDS_DIR.exists():
        return []
    out = []
    for p in sorted(config.METHOD_CARDS_DIR.glob("*.json")):
        out.append(MethodCard.from_dict(json.loads(p.read_text(encoding="utf-8"))))
    return out


def _top_papers(n: int) -> list[Paper]:
    """Top-n papers by citation_count from the data-layer corpus."""
    papers = config.load_papers()
    return sorted(papers, key=lambda p: p.citation_count, reverse=True)[:n]


def main() -> None:
    """CLI entry point: extract top-N method cards, or sample cached ones."""
    ap = argparse.ArgumentParser(description="Extract method cards from cached papers via LLM.")
    ap.add_argument("--top", type=int, default=50, help="extract for top-N papers by citations")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    ap.add_argument("--dry-run", action="store_true", help="print token estimate, no API call")
    ap.add_argument("--sample", type=int, help="print N random cached cards, no extraction")
    args = ap.parse_args()

    if args.sample is not None:
        _print_cards(_load_cached_cards(), args.sample)
        return

    papers = _top_papers(args.top)
    est = sum(_estimate_input_tokens(build_prompt(p.abstract or "", p.title)) for p in papers)
    print(
        f"Selected top {len(papers)} papers by citation_count.\n"
        f"Estimated input tokens: ~{est:,} (~{est // max(len(papers), 1):,}/paper, "
        f"output not counted). Model: {config.LLM_MODEL}"
    )

    if args.dry_run:
        print("[dry-run] no API call made.")
        return

    if not config.LLM_API_KEY:
        print("LLM_API_KEY not set. Add it to .env (DeepSeek/OpenAI-compatible).", file=sys.stderr)
        sys.exit(1)

    reply = input("Proceed? [y/N] ").strip().lower()
    if reply != "y":
        print("Aborted; no API calls made.")
        return

    extractor = MethodCardExtractor(config.LLM_API_KEY, config.LLM_BASE_URL, config.LLM_MODEL)
    cards = extractor.extract_batch(papers, force=args.force)
    print(f"\nExtracted/loaded {len(cards)}/{len(papers)} cards.")
    _print_field_rates(cards)
    print("\n--- 5 random cards for spot-check ---")
    _print_cards(cards, 5)


if __name__ == "__main__":
    main()
