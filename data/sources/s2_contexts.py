"""Semantic Scholar citation-context adapter (owner A) — fetch *why* a paper is cited.

Resolves a corpus paper (OpenAlex W-id) to an S2-queryable id via its DOI/MAG, then pulls the
paper's *reference* contexts — the sentences where it cites each reference — plus Semantic
Scholar's own citation-intent labels. Results cache per-paper under ``data/cache/s2_contexts/``
so the rate-limited crawl is resumable.

Each record is shaped to drop straight into ``nlp.citation_intent`` (``context`` + ``intents``)
and carries the cited paper's external ids so the graph annotator can match it back to a corpus
edge. See ``nlp/citation_intent/annotate_graph.py`` and ``nlp/HANDOFF.md``.

CLI: ``uv run python -m data.sources.s2_contexts <W-id> [--force]``
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any

import requests

from schemas import Paper
from spike import config

S2_API = "https://api.semanticscholar.org/graph/v1"
S2_REF_FIELDS = "contexts,intents,isInfluential,citedPaper.externalIds,citedPaper.title"
S2_PAGE_LIMIT = 1000
S2_REQUEST_DELAY = float(os.getenv("S2_REQUEST_DELAY", "1.1"))
CONTEXTS_DIR = config.CACHE_DIR / "s2_contexts"


def normalize_doi(doi: str | None) -> str | None:
    """Strip the URL prefix and lowercase a DOI for cross-source matching (None-safe)."""
    if not doi:
        return None
    bare = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip().lower()
    return bare or None


def s2_paper_id(paper: Paper) -> str | None:
    """Best Semantic Scholar lookup id for a corpus paper: DOI first, then MAG."""
    doi = normalize_doi(paper.source_ids.get("doi") or paper.external_ids.get("doi"))
    if doi:
        return f"DOI:{doi}"
    mag = paper.external_ids.get("mag")
    if mag:
        return f"MAG:{mag}"
    return None


def _headers() -> dict[str, str]:
    """Send the S2 API key when configured (raises the shared rate limit)."""
    return {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}


def _get(url: str, params: dict, max_retries: int = 5) -> dict:
    """GET an S2 JSON endpoint with polite exponential backoff on rate limits (429)."""
    delay = 2.0
    for _ in range(max_retries):
        resp = requests.get(url, params=params, headers=_headers(), timeout=30)
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"S2 rate limit: exhausted retries for {url}")


def _parse_references(items: list[dict]) -> list[dict[str, Any]]:
    """Turn raw S2 reference items into classify-ready, match-ready context records."""
    records: list[dict[str, Any]] = []
    for item in items:
        contexts = [c for c in (item.get("contexts") or []) if c]
        intents = [i for i in (item.get("intents") or []) if i]
        cited = item.get("citedPaper") or {}
        ext = cited.get("externalIds") or {}
        records.append(
            {
                "context": contexts[0] if contexts else "",
                "contexts": contexts,
                "intents": intents,
                "is_influential": bool(item.get("isInfluential")),
                "cited_doi": normalize_doi(ext.get("DOI")),
                "cited_mag": str(ext["MAG"]) if ext.get("MAG") else None,
                "cited_arxiv": ext.get("ArXiv"),
                "cited_corpus_id": ext.get("CorpusId"),
                "cited_s2_id": cited.get("paperId"),
                "cited_title": cited.get("title"),
            }
        )
    return records


def _fetch_raw(s2_id: str) -> list[dict]:
    """Page through a paper's references on the S2 Graph API (1000/page, polite delay)."""
    items: list[dict] = []
    offset = 0
    while True:
        data = _get(
            f"{S2_API}/paper/{s2_id}/references",
            {"fields": S2_REF_FIELDS, "limit": S2_PAGE_LIMIT, "offset": offset},
        )
        batch = data.get("data") or []
        items.extend(batch)
        nxt = data.get("next")
        if not batch or nxt is None:
            break
        offset = nxt
        time.sleep(S2_REQUEST_DELAY)
    return items


@lru_cache(maxsize=1)
def _corpus_s2_ids() -> dict[str, str]:
    """Map corpus paper_id -> S2 lookup id, built once from papers.json (consume only)."""
    raw = json.loads(config.PAPERS_JSON.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else list(raw.values())
    out: dict[str, str] = {}
    for record in records:
        paper = Paper.from_dict(record)
        s2_id = s2_paper_id(paper)
        if s2_id:
            out[paper.paper_id] = s2_id
    return out


def fetch_citation_contexts(
    paper_id: str, *, s2_id: str | None = None, force: bool = False
) -> list[dict]:
    """Return reference-context records for `paper_id` (text snippet + S2 intent fields).

    `paper_id` is a corpus OpenAlex W-id; its S2 lookup id is resolved from papers.json unless
    `s2_id` is supplied. Results cache to data/cache/s2_contexts/{paper_id}.json (set `force`
    to refetch). Papers with no resolvable DOI/MAG cache an empty list.
    """
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = CONTEXTS_DIR / f"{paper_id.replace('/', '_')}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    if s2_id is None:
        s2_id = _corpus_s2_ids().get(paper_id)
    records = _parse_references(_fetch_raw(s2_id)) if s2_id else []
    cache.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main() -> None:
    """CLI: fetch and preview one paper's reference contexts."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_id", help="corpus OpenAlex W-id")
    parser.add_argument("--s2-id", help="override the resolved S2 lookup id (e.g. DOI:10.x/y)")
    parser.add_argument("--force", action="store_true", help="ignore the on-disk cache")
    args = parser.parse_args()

    records = fetch_citation_contexts(args.paper_id, s2_id=args.s2_id, force=args.force)
    with_ctx = sum(1 for r in records if r["context"])
    print(f"{len(records)} reference records for {args.paper_id} ({with_ctx} with context text)")
    for record in records[:5]:
        tag = ",".join(record["intents"]) or "no-intent"
        print(f"- [{tag}] {record['cited_doi']}: {record['context'][:120]}")


if __name__ == "__main__":
    main()
