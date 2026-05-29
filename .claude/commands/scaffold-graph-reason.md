# /scaffold-graph-reason

You are scaffolding the **citation graph multi-hop reasoning** layer. 
This is the second half of the project's differentiation story (the 
first being method-card field matching from /scaffold-retrieval).

Read CLAUDE.md, docs/spike-results.md, and the HANDOFF.md files for 
data, nlp, retrieval, eval. spike/ is frozen — DO NOT MODIFY. Consume 
the citation graph (citation_graph.pkl), method cards 
(data/cache/method_cards/), and citation intents (when available from 
nlp.citation_intent).

## Goal
Implement three semantically-meaningful multi-hop graph queries that 
turn the raw citation DAG into evidence the user can read:

1. **"Find methodological ancestors"** — given a paper, trace backward 
   along method-intent citations to find the earlier work whose 
   mechanisms this paper builds on.
2. **"Find opposing work"** — find papers that this paper cites OR is 
   cited by with comparison/contrast intent.
3. **"Find cross-domain same-mechanism"** — find papers with high 
   method-card similarity but applied to a different sub-area/domain.

Each query returns not just a list, but a **traceable path** (the 
sequence of citations + intents that led to each result), so the UI 
can show "why this paper was returned."

## Branch
`feat/scaffold-graph-reason` (create from main).

## What to build

### 1. Module skeleton — `retrieval/graph_reason.py`
Single file is fine (graph reasoning is conceptually unified). 
Three public functions + supporting helpers.

### 2. Common types
```python
from dataclasses import dataclass

@dataclass
class GraphPath:
    """A trace through the citation graph explaining why a paper was returned."""
    nodes: list[str]              # paper_ids in path order, source → target
    edge_intents: list[str]       # intent for each edge (background/method/comparison)
    score: float                  # confidence / strength of this path
    explanation: str              # human-readable: "X extends Y's GCN backbone (method-citation)"

@dataclass  
class ReasoningResult:
    paper_id: str
    score: float
    paths: list[GraphPath]        # ranked, top-3 paths max
```

### 3. Function 1: `find_ancestors(paper_id, max_hops=3, k=10) -> list[ReasoningResult]`
**Semantics**: starting from `paper_id`, traverse OUTGOING edges 
(this paper cites X) preferentially weighted by "method" intent. 
Walk up to `max_hops`. Score each found paper by:
- Path length (shorter = stronger ancestor)
- Citation intent weight along the path (method > background > comparison)
- Method-card similarity (if available) — ancestor papers should share 
  task or backbone with the query paper

**Implementation**:
- Load `citation_graph.pkl` (NetworkX DiGraph)
- Use BFS from `paper_id`, following outgoing edges only
- For each visited node, look up edge intent (from nlp.citation_intent 
  classifier if available, else default to "background" weight)
- Intent weights: method=1.0, background=0.5, comparison=0.2
- Score = sum over path of (intent_weight / hop_distance) × method_card_similarity
- Build `GraphPath` with explanation like:
  `"LightGCN cites NGCF (method-intent) → NGCF cites GC-MC (method-intent), 
  both share GCN-based CF backbone"`
- Return top-k papers, each with up to 3 best paths

### 4. Function 2: `find_opposing(paper_id, k=10) -> list[ReasoningResult]`
**Semantics**: papers that this one compares against, OR that compare 
against it. The "opposition" is in *intent*, not in content disagreement.

**Implementation**:
- One-hop neighbors in both directions (cites + cited-by)
- Keep only edges with intent="comparison"
- Score = citation strength × method-card distance (we want papers 
  *contrasted* — high comparison signal + different methods, not 
  similar methods that happen to be compared)
- Explanation: `"Cited by SimGCL as baseline-to-beat (comparison-intent), 
  different backbone (LightGCN vs SimGCL+contrastive)"`

### 5. Function 3: `find_cross_domain_same_mechanism(paper_id, k=10) -> list[ReasoningResult]`
**Semantics**: papers using a similar method but applied to a different 
sub-area. This is the cleanest demo of the project's "mechanism > topic" 
claim.

**Implementation**:
- This is NOT primarily a graph query — it's a method-card query with 
  a domain filter. Reuse `MethodCardMatcher` from /scaffold-retrieval.
- Get top-30 papers by method-card similarity to query paper
- Filter to those with DIFFERENT sub-area than query paper. Sub-area 
  determination: heuristic from title/abstract keywords, OR if the 
  corpus is built with sub-area metadata (data/corpus.py records 
  which query a paper came from), use that.
- The "graph" part: optionally validate via citation graph that these 
  are independent works (not just one citing the other in same domain)
- Explanation: `"Same backbone family (LightGCN) but applied to 
  cross-domain rec instead of single-domain CF"`

### 6. Sub-area inference helper
If `data/corpus.py` records sub-area per paper (which query it came 
from), use it directly. Otherwise, implement a simple keyword classifier:
```python
SUB_AREA_KEYWORDS = {
    "contrastive": ["contrastive", "self-supervised", "InfoNCE"],
    "cross-domain": ["cross-domain", "transfer learning", "domain adaptation"],
    "knowledge-graph": ["knowledge graph", "KG", "knowledge-aware"],
    "sequential": ["sequential", "session-based", "next-item"],
    "social": ["social", "trust", "influence"],
    "collaborative-filtering": ["collaborative filtering", "matrix factorization", "implicit feedback"],
}

def infer_sub_area(paper: Paper) -> str:
    """Return best-matching sub-area or 'other'."""
    # TODO(C): replace with corpus-recorded sub-area when data/corpus.py supports it
```
Leave a TODO(C) note that this should be replaced with corpus-recorded 
sub-area once data layer provides it.

### 7. CLI — `retrieval/graph_reason.py`
uv run python -m retrieval.graph_reason 
--paper-id "W2741809807" 
--query {ancestors|opposing|cross-domain} 
--k 10

Prints results with path traces, human-readable. Example output:
Find ancestors of: "LightGCN: Simplifying and Powering Graph Convolution..."

(score 0.84) NGCF: Neural Graph Collaborative Filtering [2019]
Path: LightGCN → NGCF (method-citation, hop=1)
Why: shared backbone (GCN-based CF), LightGCN explicitly simplifies NGCF
(score 0.67) GC-MC: Graph Convolutional Matrix Completion [2017]
Path: LightGCN → NGCF (method) → GC-MC (method, hop=2)
Why: NGCF builds on GC-MC's graph convolution framework


### 8. Handoff document — `retrieval/HANDOFF.md` (append to existing)
This module is part of retrieval/, so APPEND a new section to the 
existing HANDOFF.md from /scaffold-retrieval. Don't overwrite.

New section "## Graph reasoning":
- **Three queries available**: ancestors, opposing, cross-domain
- **Intent weights are tunable** (method=1.0, background=0.5, 
  comparison=0.2) — TODO(C): tune via subjective evaluation on 5 known 
  queries, no clean nDCG signal here
- **find_opposing depends on citation intent classifier** — if B's 
  SciCite model is not ready, opposing query degrades to "any cited-by 
  with high method-card distance" (fallback works but weaker)
- **Sub-area inference is keyword-based**; should be replaced by 
  data/corpus.py recording sub-area per paper when fetched
- **Known limitation**: graph reasoning quality depends entirely on 
  citation graph density. Current 76-paper graph has |E|=174 — works. 
  After scale-up to 500 papers, may need to filter to higher-quality 
  edges (e.g., drop background-intent edges in dense regions).

### 9. Tests — `tests/test_graph_reason.py`
- `test_find_ancestors_returns_paths_in_correct_direction`: 
  build a 3-node toy graph A→B→C, find_ancestors(A) returns B then C
- `test_find_ancestors_respects_max_hops`: max_hops=1 returns only 
  direct citations
- `test_find_opposing_filters_to_comparison_intent`: build graph with 
  mixed intents, only comparison edges returned
- `test_find_cross_domain_filters_by_sub_area`: 3 papers same sub-area 
  + 2 different, cross-domain query returns only the 2
- `test_path_explanation_includes_intent_label`: returned GraphPath 
  has "method-citation" or "comparison-intent" in explanation string

Use small in-memory NetworkX graphs and mock method cards. **Do NOT 
load the real graph in tests.**

### 10. Reality check
After implementation:
1. Pick a well-known paper from the corpus (e.g., search for "LightGCN" 
   via retrieval.dense and grab its paper_id)
2. Run all three queries on it:
uv run python -m retrieval.graph_reason --paper-id <id> --query ancestors --k 5
uv run python -m retrieval.graph_reason --paper-id <id> --query opposing --k 5
uv run python -m retrieval.graph_reason --paper-id <id> --query cross-domain --k 5
3. Report the actual top results + path explanations to human reviewer

This is the demo content for answering "show me your graph reasoning" 
at defense time. If outputs are nonsensical (e.g., ancestors returns 
papers FROM 2024 for a 2020 paper), STOP and diagnose.

## Acceptance criteria
- All three functions implemented and callable via CLI
- For at least one test paper in the corpus, ancestors query returns 
  results with non-empty path explanations
- find_opposing works WITH citation intents OR via fallback (intent 
  classifier may not exist yet — that's fine, use sensible default)
- find_cross_domain returns results from different sub-areas than the 
  query paper
- `pytest tests/test_graph_reason.py` passes
- `retrieval/HANDOFF.md` updated with "## Graph reasoning" section
- `ruff check .` and `black --check .` clean

## Rules
1. **Plan mode FIRST.** Show plan, wait for "go".
2. **Do NOT modify spike/, data/, nlp/, eval/.** Read graph + method 
   cards + intents from their public outputs.
3. **One branch: feat/scaffold-graph-reason.** Commit per function.
4. **Citation intent classifier may not exist yet** (depends on 
   /scaffold-nlp + later SciCite work). Use the fallback path 
   (default intent = "background") and label this clearly in code 
   + handoff.
5. **If the citation graph is too sparse** to support reasoning (e.g., 
   most papers have 0 outgoing edges in current corpus), STOP and tell 
   me. We may need /scaffold-data to expand corpus first.
6. **If reality-check outputs are nonsensical** (wrong direction, 
   anachronistic ancestors, irrelevant cross-domain results), STOP. 
   Do not paper over with rerankers — this is the differentiation 
   story, must work cleanly.
