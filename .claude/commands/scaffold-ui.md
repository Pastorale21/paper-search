# /scaffold-ui

You are scaffolding the **UI layer** of the paper-search project. This 
is the final integration point — every previous module's value becomes 
visible (or invisible) here.

Read CLAUDE.md, docs/spike-results.md, and ALL HANDOFF.md files. spike/ 
remains frozen (the spike's ui/app.py is a working reference — copy 
ideas, don't import). Consume the public APIs of data/, nlp/, retrieval/, 
eval/.

## Goal
Replace the spike's single-page UI with a 4-tab Streamlit app that 
demonstrates every piece of the system:
1. **Search** — semantic + hybrid retrieval with "reason tags" 
   explaining why each result was returned
2. **Method Card** — structured view of a paper's task/backbone/loss/
   key_idea, plus "find similar mechanism" button
3. **Citation Graph** — interactive PyVis graph with 3 reasoning 
   query buttons (ancestors / opposing / cross-domain)
4. **Related Work Draft** — paste your own idea/abstract → generate 
   a related-work paragraph with citations

D will own this module after handoff; your job is to make Tabs 1-3 
fully functional with real data and Tab 4 functional with a 
clearly-marked LLM-prompt that D can iterate on.

## Branch
`feat/scaffold-ui` (create from main).

## What to build

### 1. App skeleton — `ui/`
Replace existing `ui/app.py` (which is spike-era). Structure:
ui/
├── app.py                  # entry point, sets up multi-page nav
├── pages/
│   ├── 1_🔍_Search.py
│   ├── 2_📋_Method_Card.py
│   ├── 3_🕸_Citation_Graph.py
│   └── 4_✍_Related_Work.py
├── components/
│   ├── init.py
│   ├── paper_card.py       # reusable paper display widget
│   ├── reason_tags.py      # render signal_breakdown as tags
│   └── graph_view.py       # PyVis wrapper
└── api.py                  # thin wrapper around data/nlp/retrieval imports

Streamlit auto-detects `pages/*.py` files; `app.py` is the landing page.

### 2. `ui/api.py` — single point of integration
Centralize all imports from other modules here. Other UI files import 
ONLY from `ui.api`. This way if a downstream interface changes, only 
`api.py` needs updating.

```python
@st.cache_resource
def get_hybrid_retriever() -> HybridRetriever: ...

@st.cache_resource
def get_method_card_matcher() -> MethodCardMatcher: ...

@st.cache_data
def load_papers() -> dict[str, Paper]: ...

@st.cache_data
def load_method_card(paper_id: str) -> MethodCard | None: ...

def search(query: str, mode: str, k: int) -> list[dict]: 
    """Returns enriched results with paper, score, signal_breakdown, method_card."""

def find_ancestors(paper_id: str, k: int) -> list[ReasoningResult]: ...
def find_opposing(paper_id: str, k: int) -> list[ReasoningResult]: ...
def find_cross_domain(paper_id: str, k: int) -> list[ReasoningResult]: ...
```

### 3. `ui/app.py` — landing page
Minimal landing:
- Title: "GNN-RecSys Paper Search"
- One-paragraph description of the system's differentiation 
  (mechanism-level matching + graph reasoning)
- Quick stats from cached artifacts: corpus size, method card 
  coverage, graph edges
- Sidebar nav handled by Streamlit automatically (it picks up pages/)

### 4. `pages/1_🔍_Search.py` — Search tab
- Mode toggle: 短查询 / 粘贴论文摘要 (st.radio)
- Top-K slider (default 10)
- Method selector: dense / bm25 / dense_rerank / hybrid (default hybrid)
- Query input (st.text_input or st.text_area depending on mode)
- Search button → calls `api.search()`
- Results: vertical list of paper cards. Each card shows:
  - Title + year + citation_count
  - **Reason tags** (from signal_breakdown): colored badges like 
    `dense: 0.95 / method_match: 0.81 / bm25: 0.12`
  - Score (final hybrid score)
  - Abstract (collapsed by default, expander)
  - "View method card" link → switches to Tab 2 with this paper selected
  - "Show in graph" link → switches to Tab 3 with this paper as center

**Reason tags are the visible payoff of /scaffold-retrieval** — make 
them prominent. Use `components/reason_tags.py` to render them as 
colored chip-style badges.

### 5. `pages/2_📋_Method_Card.py` — Method Card tab
- Paper selector: st.selectbox over papers (search by title to filter)
  OR accept `?paper_id=...` URL param for deep-linking from Tab 1
- Layout: 2 columns
  - Left: structured method card (task, input, output, backbone, loss, 
    key_idea highlighted, datasets as chips, metrics as chips)
  - Right: original abstract
- Below: **"Find papers with similar mechanism" button** → calls 
  `MethodCardMatcher.match()` with this paper as query, displays top-10 
  in a compact list with the field-level match breakdown
  (e.g., `backbone: 0.91, loss: 0.95, key_idea: 0.78`)
- If method card not extracted yet for this paper: show clear message 
  + button "Extract now" (calls nlp.method_card.extractor for one paper)

### 6. `pages/3_🕸_Citation_Graph.py` — Citation Graph tab
- Paper selector (same pattern as Tab 2)
- Three query buttons:
  - 🌳 "Find methodological ancestors"
  - ⚔️ "Find opposing work"
  - 🌐 "Find cross-domain same mechanism"
- On click → call corresponding `api.find_*()` function
- Results display:
  - Top: PyVis graph showing the center paper + result papers + 
    citation edges along the paths. Edge colors by intent (gray=background, 
    blue=method, orange=comparison). Highlight the queried center node.
  - Below: list of returned papers with their `GraphPath.explanation` 
    text shown prominently (this is the "why" the user came for)

**PyVis integration**: use the spike's pattern:
```python
net.write_html("tmp_graph.html", notebook=False)
with open("tmp_graph.html") as f:
    components.html(f.read(), height=600)
```
Set `cdn_resources='in_line'` to avoid the lib/ folder pollution issue 
that hit the spike (mentioned in CLAUDE.md's Do NOT section).

### 7. `pages/4_✍_Related_Work.py` — Related Work Draft tab
This is the most "AI feature" tab and the highest-variance to 
implement. Build a **functional but obviously-iterable** version:

- Large st.text_area: "Paste your paper idea or draft abstract"
- Slider: number of citations to use (5-15)
- Slider: target paragraph length (150-400 words)
- "Generate" button → 
  1. Treat input as paper-as-query, retrieve top-N candidate papers 
     via hybrid retriever
  2. Load method cards for each
  3. Call LLM (reuse `nlp.method_card.extractor`'s client) with a 
     prompt that includes: user's input + retrieved papers' titles + 
     key_idea fields + asks for a coherent related-work paragraph 
     with `[N]` style citations matched to a returned bibliography list
  4. Parse LLM output, render paragraph + clickable citations + 
     expanded references list at bottom
- Below the generated text: "Fact-check" expander showing for each 
  citation, the source paper + a key sentence from its abstract 
  (helps D iterate on factuality later)

**Prompt template lives in `ui/related_work_prompt.py`** — single file, 
clearly marked with `# TODO(D): iterate on this prompt`. Don't bury it 
inside the page logic.

This tab's quality WILL be mediocre on day 1 — that's expected, 
prompt engineering is D's job. Make the wiring clean so D's iteration 
loop is fast (change prompt → reload page → see new output).

### 8. `components/paper_card.py`
Reusable function `render_paper_card(paper, score=None, signal_breakdown=None, show_actions=True)`:
- Displays title, year, citations, abstract (collapsible)
- If score: show prominently
- If signal_breakdown: render reason tags via reason_tags component
- If show_actions: "Method card" + "Citation graph" deep-link buttons

### 9. `components/reason_tags.py`
Function `render_reason_tags(signal_breakdown: dict)`:
- Render each signal as a colored chip badge
- Color scheme: dense=blue, bm25=gray, method_match=green, 
  cross_encoder=purple
- Show score next to label, e.g., `[method_match: 0.81]`
- Use `st.markdown` with inline HTML/CSS or Streamlit's native badges 
  if available — keep it visually clean

### 10. `components/graph_view.py`
Function `render_graph(center_node_id, paths: list[GraphPath], height=600)`:
- Build PyVis Network from paths
- Color nodes: center=red, others=blue
- Color edges by intent: background=gray, method=blue, comparison=orange
- Add hover tooltips with paper title
- Render inline via st.components.html

### 11. Handoff document — `ui/HANDOFF.md`
Write for D. Include:
- **You are**: D (UI + eval owner)
- **Current state**:
  - Tabs 1-3 fully functional against real data
  - Tab 4 wired end-to-end but prompt is v0 (your iteration target)
  - All retrieval/reasoning calls go through `ui/api.py` — single 
    surface to maintain
- **Your work, prioritized**:
  - P0: Iterate on Tab 4's related-work prompt (`ui/related_work_prompt.py`). 
    Goal: paragraph reads coherently, citations are factually 
    supported by retrieved papers, no hallucinated paper names. 
    Test with 5-10 of your own idea-stubs.
  - P0: Visual polish. Current UI is functional-not-pretty. Improve:
    color consistency, spacing, loading states (st.spinner), error 
    states ("no method card available"), empty states
  - P1: Mobile-friendly layout (or at least don't break on narrow screens)
  - P1: Save/share results — let user copy a permalink with 
    query+method in URL params
  - P2: Demo mode — pre-populate Tab 4 with a known good example, 
    pre-cache its retrieval, so live demo is snappy
- **Interface contracts (do NOT change without team review)**:
  - All UI → backend calls go through `ui/api.py`. Other files 
    import ONLY from there.
  - Streamlit page filenames (numbers control nav order)
- **Known issues / gotchas**:
  - PyVis still writes some files to cwd despite cdn_resources='in_line' 
    in some cases — gitignore `tmp_graph.html` and `lib/` if reappears
  - First page load is slow (model loading); subsequent loads use 
    @st.cache_resource — don't accidentally bust the cache
  - Streamlit pinned to 1.57.0 (see CLAUDE.md). `st.components.v1.html` 
    deprecation is YOUR migration task before end of project
  - Tab 4 LLM calls cost money — add a hard daily limit if exposed publicly
- **Tools to learn**:
  - Streamlit multi-page: docs.streamlit.io/develop/concepts/multipage-apps (~15 min)
  - Streamlit caching: docs.streamlit.io/develop/concepts/architecture/caching (~10 min)
  - PyVis basics: pyvis.readthedocs.io (~10 min, mostly already done in spike)

### 12. Tests
UI testing is awkward; do the minimum:
- `tests/test_ui_api.py`:
  - `test_api_search_returns_enriched_results`: mock HybridRetriever, 
    verify api.search wraps output with paper + method_card
  - `test_api_handles_missing_method_card`: paper without card → 
    enriched result has method_card=None, not crash
- Skip Streamlit page tests (too brittle for course project scope)

### 13. Reality check
After implementation:
1. Run `uv run streamlit run ui/app.py`
2. In each tab, do ONE real interaction and screenshot the result:
   - Tab 1: search "graph contrastive learning" with hybrid method, 
     screenshot the result list showing reason tags
   - Tab 2: pick LightGCN (or any paper in corpus), screenshot method 
     card view + similar-mechanism results
   - Tab 3: pick the same paper, click "Find ancestors", screenshot 
     the graph + path explanations
   - Tab 4: paste a short test abstract, generate, screenshot the 
     paragraph + citations
3. Report any tab that doesn't work end-to-end. THESE FOUR SCREENSHOTS 
   are the project's demo material — they must look reasonable.

## Acceptance criteria
- `uv run streamlit run ui/app.py` launches without error
- All 4 tabs accessible via sidebar nav
- Tabs 1-3 perform real retrieval/reasoning against actual cached data
- Tab 4 generates a paragraph (quality v0 — D's iteration target)
- Reason tags visible on Tab 1 results
- PyVis graph renders on Tab 3 with intent-colored edges
- `tests/test_ui_api.py` passes
- `ui/HANDOFF.md` exists
- `ruff check .` and `black --check .` clean
- Reality check screenshots delivered (or described in detail if image 
  upload not possible)

## Rules
1. **Plan mode FIRST.** Show plan, wait for "go".
2. **Do NOT modify spike/, data/, nlp/, retrieval/, eval/.** Consume 
   their public interfaces only.
3. **One branch: feat/scaffold-ui.** Commit per tab/page.
4. **All backend imports go through ui/api.py.** Other UI files do 
   NOT import directly from retrieval/, nlp/, etc.
5. **If Tab 4 LLM output is garbage** (no citations, hallucinated 
   papers), that's EXPECTED for v0 — leave the v0 prompt with a TODO(D) 
   note and move on. Don't spend cycles tuning a prompt that D will 
   own and iterate.
6. **If any tab fails the reality check**, STOP and report. Don't 
   ship a non-working demo.
7. **lib/ and tmp_graph.html in cwd** — confirm these don't appear, 
   or add to .gitignore if they do. The spike already hit this.
