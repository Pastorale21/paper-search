"""OpenAlex ingestion adapter (free, no key): abstracts via inverted index, edges via refs.

Ported and generalized from spike/fetch.py. Reads endpoint config from spike.config (which also
sets the macOS libomp guard transitively). Populates Paper.source_ids with normalized openalex +
doi keys; keeps the raw OpenAlex ids dict on Paper.external_ids for backward compatibility.
"""

from __future__ import annotations

import time

import requests

from schemas import Paper
from spike import config


def _get(url: str, params: dict, headers: dict | None = None, max_retries: int = 5) -> dict:
    """GET a JSON endpoint with polite exponential backoff.

    Retries on rate limits (429) AND transient network errors (read timeouts, connection
    resets) — a multi-hundred-call crawl (e.g. corpus build / seed merge) must not abort on a
    single slow OpenAlex response.
    """
    delay = 2.0
    for _ in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers or {}, timeout=30)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"exhausted retries (rate limit or network) for {url}")


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


def _source_ids(ids: dict) -> dict[str, str]:
    """Normalize OpenAlex's raw `ids` dict into the canonical source_ids map."""
    out: dict[str, str] = {}
    oa = _short_id(ids.get("openalex"))
    if oa:
        out["openalex"] = oa
    doi = ids.get("doi")
    if doi:
        out["doi"] = doi.replace("https://doi.org/", "")
    return out


def _oa_to_paper(item: dict) -> Paper:
    authors = [
        a["author"]["display_name"]
        for a in (item.get("authorships") or [])
        if a.get("author", {}).get("display_name")
    ]
    refs = [_short_id(r) for r in (item.get("referenced_works") or [])]
    ids = item.get("ids") or {}
    return Paper(
        paper_id=_short_id(item["id"]),
        title=item.get("title") or "",
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        year=item.get("publication_year"),
        authors=authors,
        citation_count=item.get("cited_by_count") or 0,
        external_ids=ids,
        references=[r for r in refs if r],
        citations=[],  # OpenAlex exposes cited_by_count only, not the citing-work list
        source_ids=_source_ids(ids),
    )


def fetch_works(query: str, n: int) -> list[Paper]:
    """Search OpenAlex for `n` works matching `query`; return all parsed Papers (no filtering)."""
    papers: list[Paper] = []
    cursor = "*"
    while len(papers) < n and cursor:
        params = {
            "search": query,
            "per-page": min(n - len(papers), 200),
            "select": config.OPENALEX_SELECT,
            "cursor": cursor,
        }
        if config.OPENALEX_MAILTO:
            params["mailto"] = config.OPENALEX_MAILTO
        data = _get(config.OPENALEX_URL, params)
        batch = [_oa_to_paper(it) for it in (data.get("results") or []) if it.get("id")]
        if not batch:
            break
        papers.extend(batch)
        cursor = (data.get("meta") or {}).get("next_cursor")
    print(f"[openalex] query={query!r} fetched {len(papers)} works")
    return papers
