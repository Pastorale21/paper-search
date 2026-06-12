"""Gold-set loader + title resolver for evaluating retrieval against curated answers.

The title resolver maps gold paper titles (often short canonical abbreviations like ``NGCF``
or ``GC-MC``) to ``paper_id`` strings present in ``data/cache/papers.json``. Matching is
five-tiered (exact → colon-prefix → alias-map → substring → difflib fuzzy ≥ 0.85) because
most abbreviations don't appear verbatim in OpenAlex titles. The alias map is
hand-maintained — see eval/HANDOFF.md for the extension protocol.

CLI: ``uv run python -m eval.gold_set --check``
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from schemas import Paper

GOLD_SET_PATH = Path(__file__).resolve().parent / "gold_set.json"
PAPERS_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "papers.json"
MIN_RESOLUTION_RATE = 0.60  # STOP gate per the scaffold rules

# Hand-maintained alias map: abbreviation (normalized) -> substring of canonical paper
# title (normalized). The resolver looks up the substring against the corpus's normalized
# title index. Keep entries lowercase + punctuation-stripped — the normalizer does the rest.
# See eval/HANDOFF.md for the extension protocol when D adds new gold papers.
DEFAULT_ALIASES: dict[str, str] = {
    # LightGCN-family CF (seed papers; titles don't contain the acronym)
    "ngcf": "neural graph collaborative filtering",
    "gcmc": "graph convolutional matrix completion",
    "gc mc": "graph convolutional matrix completion",
    "lrgccf": "revisiting graph based collaborative filtering",
    "lr gccf": "revisiting graph based collaborative filtering",
    "dgcf": "disentangled graph collaborative filtering",
    "ultragcn": "ultragcn",
    # Contrastive seed papers
    "sgl": "self supervised graph learning for recommendation",
    "ncl": "neighborhood enriched contrastive learning",
    "hccf": "hypergraph contrastive collaborative filtering",
    # title: "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning..."
    "simgcl": "simple graph contrastive learning",
    "xsim gcl": "extremely simple graph contrastive learning",
    "xsimgcl": "extremely simple graph contrastive learning",
    "lightgcl": "lightgcl simple yet effective graph contrastive learning",
    "dccf": "disentangled contrastive collaborative filtering",
    # Knowledge-graph seed paper(s)
    "kgin": "learning intents behind interactions",
    # title: "Knowledge Graph Convolutional Networks for Recommender Systems"
    "kgcn": "kgcn knowledge graph convolutional networks for recommender systems",
    "cke": "collaborative knowledge base embedding",
    "ckan": "collaborative knowledge aware attentive network",
    "kgnnls": "knowledge graph neural networks with label smoothness",
    "kgnn ls": "knowledge graph neural networks with label smoothness",
    # Cross-domain seed paper
    "bitgcf": "bi directional transfer graph",
    "bi tgcf": "bi directional transfer graph",
    "ccdr": "contrastive cross domain recommendation",
    "disencdr": "disentangled representations for cross domain recommendation",
    "disen cdr": "disentangled representations for cross domain recommendation",
    # DDTCDR canonical title is "Deep Dual Transfer Cross Domain Recommendation" (WSDM'20);
    # colon-prefix already catches "DDTCDR:" titles, this is the no-colon substring fallback.
    "ddtcdr": "deep dual transfer cross domain recommendation",
    "ddt cdr": "deep dual transfer cross domain recommendation",
    "ppgn": "preference propagation graphnet",
    # real PTUPCDR paper (W3209185641) added via data.corpus --merge-seeds; align to its title
    "ptupcdr": "personalized transfer of user preferences for cross domain recommendation",
    "ptup cdr": "personalized transfer of user preferences for cross domain recommendation",
    # Session-based seed papers
    "gcsan": "graph contextualized self attention",
    "gc san": "graph contextualized self attention",
    "srgnn": "session based recommendation with graph neural",
    "sr gnn": "session based recommendation with graph neural",
    "s3rec": "self supervised learning for sequential recommendation",
    "s3 rec": "self supervised learning for sequential recommendation",
    "cl4srec": "contrastive learning for sequential recommendation",
    "cl4s rec": "contrastive learning for sequential recommendation",
    "tagnn": "target attentive graph neural networks",
    "gc egnn": "global context enhanced graph neural",
    "surge": "sequential recommendation with graph neural networks",
    "fgnn": "feature graph neural networks for session based recommendation",
    "lessr": "handling information loss of graph neural networks",
    # Social seed papers
    "diffnet": "a neural influence diffusion model for social recommendation",
    "mhcn": "multi channel hypergraph convolutional network for social recommendation",
    "graphrec": "graph neural networks for social recommendation",
    "sociallgn": "light graph convolution network for social recommendation",
    "social lgn": "light graph convolution network for social recommendation",
    # `_norm("DiffNet++")` collapses "++" to one "plusplus" token, so the key MUST match that
    # (the old "diffnet plus plus" key was unreachable). See test_diffnet_plusplus_alias_key.
    "diffnet plusplus": "diffnet plusplus",
    "dhcf": "dual channel hypergraph collaborative filtering",
    "hgcn": "hypergraph convolutional network for collaborative filtering",
    # Aliases below aligned to the REAL papers added via `data.corpus --merge-seeds` (verified
    # by year/venue). HyperRec/FairGo/FairGNN titles don't contain the acronym, so the bare-token
    # aliases never matched until the real paper was present.
    "hyperrec": "next item recommendation with sequential hypergraphs",
    "fairgo": "learning fair representations for recommendation a graph based perspective",
    "fairrec": "fairrec",
    "nfcf": "neural fairness collaborative filtering",
    "gfair": "gfair",
    "fairgnn": "say no to the discrimination learning fair graph neural networks",
    "metahin": "meta learning on heterogeneous information networks",
    "duorec": "contrastive learning for representation degeneration problem",
    "coserec": "contrastive self supervised sequential recommendation with robust augmentation",
    "iclrec": "intent contrastive learning for sequential recommendation",
    "gfcf": "graph filter collaborative filtering",
    "pinsage": "graph convolutional neural networks for web scale recommender systems",
    # Non-seed but verified present-via-full-expansion-title
    "gcegnn": "global context enhanced graph neural",
    "gce gnn": "global context enhanced graph neural",
    "kcgn": "knowledge aware coupled graph neural",
}


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NONALNUM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _norm(text: str | None) -> str:
    """Strip HTML tags, lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", text)
    cleaned = cleaned.replace("++", " plusplus ")
    cleaned = cleaned.replace("+", " plus ")
    cleaned = _NONALNUM_RE.sub(" ", cleaned.lower())
    return _WS_RE.sub(" ", cleaned).strip()


@dataclass
class GoldQuery:
    """One curated evaluation query with its expected top-N gold papers."""

    id: str
    mode: str  # "short" or "paper"
    text: str
    gold_titles: list[str]
    notes: str = ""
    anchor_title: str | None = None  # paper-mode only; deterministic anchor for method_match
    arxiv: str | None = None


@dataclass
class GoldSet:
    """A versioned gold set of queries (load from gold_set.json)."""

    version: str
    created: str
    queries: list[GoldQuery] = field(default_factory=list)
    owner: str = ""
    scope: str = ""

    @classmethod
    def load(cls, path: Path = GOLD_SET_PATH) -> "GoldSet":
        """Load and validate a gold_set.json file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        queries = [
            GoldQuery(
                id=q["id"],
                mode=q["mode"],
                text=q["text"],
                gold_titles=list(q["gold_titles"]),
                notes=q.get("notes", ""),
                anchor_title=q.get("anchor_title"),
                arxiv=q.get("arxiv"),
            )
            for q in raw.get("queries", [])
        ]
        return cls(
            version=raw["version"],
            created=raw["created"],
            queries=queries,
            owner=raw.get("owner", ""),
            scope=raw.get("scope", ""),
        )


@dataclass
class _ResolveHit:
    paper_id: str
    matched_title: str
    strategy: str  # "exact" | "colon-prefix" | "alias" | "substring" | "fuzzy"


class TitleResolver:
    """Map gold-set titles to ``paper_id`` via a five-tier matching cascade."""

    def __init__(
        self,
        papers: list[Paper] | list[dict],
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._aliases = aliases if aliases is not None else DEFAULT_ALIASES
        # Accept Paper objects OR dicts (the eval may load papers.json without going through
        # schemas.Paper).
        self._records: list[tuple[str, str]] = [
            (
                p.paper_id if isinstance(p, Paper) else p["paper_id"],
                p.title if isinstance(p, Paper) else (p.get("title") or ""),
            )
            for p in papers
        ]
        self._by_full: dict[str, str] = {}
        self._by_abbrev: dict[str, str] = {}
        self._all_norm_titles: list[str] = []
        self._norm_title_to_id: dict[str, str] = {}
        for pid, title in self._records:
            nt = _norm(title)
            if nt:
                self._by_full.setdefault(nt, pid)
                self._all_norm_titles.append(nt)
                self._norm_title_to_id.setdefault(nt, pid)
            if title and ":" in title:
                head = title.split(":", 1)[0]
                head_n = _norm(head)
                if head_n:
                    self._by_abbrev.setdefault(head_n, pid)
        # Public log of fuzzy + alias matches for the human reviewer to verify.
        self.fuzzy_log: list[tuple[str, str, str]] = []

    def resolve(self, title: str) -> str | None:
        """Return ``paper_id`` for the gold title, or ``None`` if no plausible match."""
        hit = self._resolve_with_strategy(title)
        return hit.paper_id if hit else None

    def _resolve_with_strategy(self, title: str) -> _ResolveHit | None:
        n = _norm(title)
        if not n:
            return None
        # 1. exact normalized title match
        if n in self._by_full:
            return _ResolveHit(self._by_full[n], n, "exact")
        # 2. colon-prefix abbreviation (e.g. "LightGCN" -> "LightGCN: Simplifying…")
        if n in self._by_abbrev:
            return _ResolveHit(self._by_abbrev[n], n, "colon-prefix")
        # 3. alias map (e.g. "NGCF" -> substring "neural graph collaborative filtering")
        alias_target = self._aliases.get(n)
        if alias_target:
            for nt in self._all_norm_titles:
                if alias_target in nt:
                    pid = self._norm_title_to_id[nt]
                    self.fuzzy_log.append((title, nt, "alias"))
                    return _ResolveHit(pid, nt, "alias")
            # A curated acronym alias that misses its target is a corpus gap, not a license
            # to substring-match another acronym-containing title such as SelfGNN -> FGNN.
            return None
        # 4. substring: gold (normalized) appears in any paper title
        if len(n) >= 4:  # avoid trivial matches like "cf"
            for nt in self._all_norm_titles:
                if n in nt:
                    pid = self._norm_title_to_id[nt]
                    self.fuzzy_log.append((title, nt, "substring"))
                    return _ResolveHit(pid, nt, "substring")
        # 5. difflib fuzzy fallback at 0.85 cutoff
        close = difflib.get_close_matches(n, self._all_norm_titles, n=1, cutoff=0.85)
        if close:
            nt = close[0]
            pid = self._norm_title_to_id[nt]
            self.fuzzy_log.append((title, nt, "fuzzy"))
            return _ResolveHit(pid, nt, "fuzzy")
        return None


def _load_corpus() -> list[dict]:
    """Read ``data/cache/papers.json`` (dicts, no schema dependency)."""
    if not PAPERS_JSON_PATH.exists():
        raise FileNotFoundError(f"{PAPERS_JSON_PATH} missing; run data.corpus first")
    raw = json.loads(PAPERS_JSON_PATH.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else list(raw.values())


def main() -> int:
    """CLI: print per-query resolution + summary; exit non-zero if rate < 60%."""
    ap = argparse.ArgumentParser(description="Check gold-set title resolution against the corpus.")
    ap.add_argument(
        "--check", action="store_true", help="resolve every gold title and print coverage"
    )
    ap.add_argument("--gold", type=Path, default=GOLD_SET_PATH, help="path to gold_set.json")
    args = ap.parse_args()
    if not args.check:
        ap.print_help()
        return 0

    gs = GoldSet.load(args.gold)
    papers = _load_corpus()
    resolver = TitleResolver(papers)

    total = 0
    resolved = 0
    print(
        f"[gold-set] {gs.version} ({gs.created}) - {len(gs.queries)} queries vs "
        f"corpus of {len(papers)} papers"
    )
    print()
    for q in gs.queries:
        marks: list[str] = []
        q_resolved = 0
        for t in q.gold_titles:
            pid = resolver.resolve(t)
            if pid:
                marks.append(f"{t} OK")
                q_resolved += 1
            else:
                marks.append(f"{t} MISS")
        total += len(q.gold_titles)
        resolved += q_resolved
        print(f"  {q.id}: {q_resolved}/{len(q.gold_titles)} resolved  ({', '.join(marks)})")
        if q.anchor_title:
            apid = resolver.resolve(q.anchor_title)
            print(f"        anchor={q.anchor_title!r} -> {apid or 'UNRESOLVED'}")

    rate = resolved / total if total else 0.0
    print()
    print(f"[summary] {resolved}/{total} = {rate:.1%} resolved")

    # Surface fuzzy / alias / substring matches so the human can verify they're sane.
    if resolver.fuzzy_log:
        print()
        print(f"[match log] {len(resolver.fuzzy_log)} non-exact matches (verify):")
        for gold_title, matched, strategy in resolver.fuzzy_log:
            print(f"  [{strategy}] {gold_title!r:<24} -> {matched[:90]!r}")

    if rate < MIN_RESOLUTION_RATE:
        print()
        print(
            f"[STOP] resolution {rate:.1%} < {MIN_RESOLUTION_RATE:.0%}; "
            f"either trim the gold set or expand the corpus."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
