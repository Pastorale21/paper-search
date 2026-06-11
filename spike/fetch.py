"""Fetch GNN-recsys papers (with citation edges) from OpenAlex (default) or Semantic Scholar."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from schemas import Paper

from . import config


def _get(url: str, params: dict, headers: dict | None = None, max_retries: int = 5) -> dict:
    """GET a JSON endpoint with polite exponential backoff on rate limits (429)."""
    delay = 2.0
    for _ in range(max_retries):
        resp = requests.get(url, params=params, headers=headers or {}, timeout=30)
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Rate limit: exhausted retries for {url}")


# --- Semantic Scholar backend ---------------------------------------------------------------


def _s2_to_paper(item: dict) -> Paper:
    refs = [r["paperId"] for r in (item.get("references") or []) if r.get("paperId")]
    cits = [c["paperId"] for c in (item.get("citations") or []) if c.get("paperId")]
    authors = [a.get("name", "") for a in (item.get("authors") or [])]
    return Paper(
        paper_id=item["paperId"],
        title=item.get("title") or "",
        abstract=item.get("abstract"),
        year=item.get("year"),
        authors=authors,
        citation_count=item.get("citationCount") or 0,
        external_ids=item.get("externalIds") or {},
        references=refs,
        citations=cits,
    )


def _fetch_s2(query: str, n: int) -> list[Paper]:
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    params = {"query": query, "limit": min(n, 100), "fields": config.S2_FIELDS}
    data = _get(config.S2_SEARCH_URL, params, headers=headers)
    return [_s2_to_paper(it) for it in (data.get("data") or []) if it.get("paperId")]


# --- OpenAlex backend -----------------------------------------------------------------------


def _short_id(oa_id: str | None) -> str | None:
    """Turn 'https://openalex.org/W123' into 'W123'."""
    return oa_id.rsplit("/", 1)[-1] if oa_id else oa_id


def _reconstruct_abstract(inv: dict | None) -> str | None:
    """Rebuild an abstract string from OpenAlex's abstract_inverted_index."""
    if not inv:
        return None
    positions = [(pos, word) for word, idxs in inv.items() for pos in idxs]
    positions.sort()
    text = " ".join(word for _, word in positions)
    return text or None


def _oa_to_paper(item: dict) -> Paper:
    authors = [
        a["author"]["display_name"]
        for a in (item.get("authorships") or [])
        if a.get("author", {}).get("display_name")
    ]
    refs = [_short_id(r) for r in (item.get("referenced_works") or [])]
    return Paper(
        paper_id=_short_id(item["id"]),
        title=item.get("title") or "",
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        year=item.get("publication_year"),
        authors=authors,
        citation_count=item.get("cited_by_count") or 0,
        external_ids=item.get("ids") or {},
        references=[r for r in refs if r],
        citations=[],  # OpenAlex exposes cited_by_count only, not the citing-work list
    )


def _fetch_openalex(query: str, n: int) -> list[Paper]:
    params = {"search": query, "per-page": min(n, 200), "select": config.OPENALEX_SELECT}
    if config.OPENALEX_MAILTO:
        params["mailto"] = config.OPENALEX_MAILTO
    data = _get(config.OPENALEX_URL, params)
    return [_oa_to_paper(it) for it in (data.get("results") or []) if it.get("id")]


# --- public API -----------------------------------------------------------------------------


def fetch_papers(query: str = config.QUERY, n: int = config.N_PAPERS) -> list[Paper]:
    """Search the configured backend for `n` papers, keeping only those with abstracts."""
    if config.DATA_SOURCE == "s2":
        papers = _fetch_s2(query, n)
    else:
        papers = _fetch_openalex(query, n)
    with_abstract = [p for p in papers if p.abstract]
    print(
        f"[fetch] source={config.DATA_SOURCE} retrieved {len(papers)} papers, "
        f"{len(with_abstract)} with abstracts (kept)"
    )
    return with_abstract


def save_papers(papers: list[Paper], path: Path = config.PAPERS_JSON) -> None:
    """Write papers to cache as JSON."""
    path.write_text(
        json.dumps([p.to_dict() for p in papers], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_papers(path: Path = config.PAPERS_JSON) -> list[Paper]:
    """Read papers from cache JSON."""
    return [Paper.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def main(force: bool = False, query: str = config.QUERY, n: int = config.N_PAPERS) -> list[Paper]:
    """Fetch + cache papers, skipping the network call if cache exists (unless force)."""
    if config.PAPERS_JSON.exists() and not force:
        papers = load_papers()
        print(f"[fetch] cache hit: {len(papers)} papers")
        return papers
    papers = fetch_papers(query, n)
    save_papers(papers)
    return papers
