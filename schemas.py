"""Shared data schemas for the paper-search system (team-owned; coordinate before changing)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class Paper:
    """A scientific paper with metadata and citation edges (Semantic Scholar-sourced)."""

    paper_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    citation_count: int = 0
    external_ids: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Paper":
        """Construct from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MethodCard:
    """Mechanism-level summary of a paper's method (skeleton; populated by nlp/ later)."""

    paper_id: str
    name: str | None = None
    problem: str | None = None
    method: str | None = None
    key_idea: str | None = None
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MethodCard":
        """Construct from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
