"""Semantic Scholar citation-context adapter (extension point for owner A — week 3).

Fetches the surrounding text of each citation so B's intent classifier can label *why* a paper
is cited. Stubbed until an S2 API key is available; see data/HANDOFF.md.
"""

from __future__ import annotations


def fetch_citation_contexts(paper_id: str) -> list[dict]:
    """Return citation-context records for `paper_id` (text snippet + intent fields)."""
    raise NotImplementedError("TODO(A): implement with S2 API key, see HANDOFF.md")
