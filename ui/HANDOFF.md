# UI Layer — Handoff

## You are
**D — UI + eval owner.** This module sits on top of every other module's public API.
When you change anything here, the demo audience sees it; when you break the wiring
between this layer and one of the backend modules, **only `ui/api.py` needs updating**
(every other UI file imports through there).

## Current state — what runs out of the box

- `ui/app.py` — landing with corpus stats + a one-paragraph differentiation pitch.
- `ui/pages/1_🔍_Search.py` — mode toggle (short/paper), method selector
  (`hybrid` default; `dense / bm25 / dense_rerank` for comparison), reason-tag chips on
  every hybrid result, deep-link buttons to Tabs 2 and 3.
- `ui/pages/2_📋_Method_Card.py` — structured method-card view + the **🔍 Find papers with
  similar mechanism** showcase. Each ranked match shows the **per-field cosines**
  (`backbone (w=0.30): 0.91`, `loss (w=0.20): 0.95`, ...) as colored badges — this is the
  *visible evidence* that mechanism-level matching ranked the paper. Don't bury this; it's
  the headline differentiation.
- `ui/pages/3_🕸_Citation_Graph.py` — three reasoning buttons in the documented order
  **🌳 Ancestors → 🌐 Cross-domain → ⚔️ Opposing** (strongest demo first, weakest fallback
  last). Opposing carries an in-UI banner that says it's running on the
  mechanism-distance fallback (no edge intents on disk yet) plus a survey-title filter on
  top of the matcher's empty-card exclusion.
- `ui/pages/4_✍_Related_Work.py` — hybrid retrieval → method-card lookup →
  `build_messages()` → LLM call → JSON parse → paragraph + references + fact-check
  expander. **Generate fires only when the user clicks it.**
- `ui/components/{paper_card, reason_tags, graph_view}.py` — reusable widgets. PyVis uses
  `cdn_resources='in_line'` and writes to `tempfile` (NOT cwd) — see Known issues.

## Hard rule: paid LLM calls are user-initiated

Tab 4's Generate button is the only paid-LLM surface in the UI. **Do not add any other
auto-firing LLM path** (e.g. a "use AI to summarize this paper" auto-load on Tab 2). The
saved feedback rule is "user runs all paid LLM calls manually" — that includes anything
the UI fires without an explicit click. Tab 2's missing-card case deliberately shows the
extractor CLI rather than a one-click button; **do not change that to a one-click button**.

## Your work, prioritized

- **P0 — iterate on `ui/related_work_prompt.py`.** v0 prompt asks for strict JSON
  (`paragraph` + `references[]`) plus a `[N]` marker convention. Goal:
  (a) paragraph reads coherently in academic English,
  (b) every `[N]` corresponds to a real retrieved paper (no hallucinations),
  (c) ends with one sentence contrasting the user's draft against the cited line.
  Test loop: change `SYSTEM_PROMPT`, reload the page, click Generate with the same input,
  eyeball the diff. The fact-check expander already shows the raw LLM response + the
  messages sent — your debug surface is built-in.
- **P0 — visual polish.** Right now Tabs are functional, not pretty. Improve color
  consistency, spacing, `st.spinner` placement on slow buttons, empty states (Tab 3 for
  foundational papers — see "OUT-sparsity" in `retrieval/HANDOFF.md`), error states (Tab 2
  for missing cards, Tab 4 for parse errors). The `st.badge` color palette in
  `reason_tags.py` and `Method_Card.py` is the spine — match it elsewhere.
- **P1 — mobile-friendly layout** (or at least: don't break on narrow screens).
- **P1 — share-link support.** Streamlit 1.57.0 has `st.query_params` — let users copy
  a URL containing query + method + paper_id so demos and feedback can deep-link.
- **P2 — demo mode.** Pre-populate Tab 4 with a known-good test abstract, pre-cache its
  retrieval, so live demo is snappy.
- **P2 — `eval/` extension.** When you add gold queries (eval P0 in `eval/HANDOFF.md`),
  remember the Q1 finding from the diagnostic: dense top-10 are all on-topic but few are
  the *canonical* comparators. Consider adding an `accepted_titles` field next to
  `gold_titles` for lenient recall metrics.

## Interface contracts (do NOT change without team review)

- **All UI → backend calls go through `ui/api.py`.** Pages and components import ONLY from
  `ui.api` (or other UI files). If you find a page importing `from retrieval.foo` or
  `from nlp.bar`, that's a bug — move the import into `ui/api.py` and re-export.
- **`ui.api.search(query, mode, method, k)`** returns `list[dict]` with keys
  `{paper, paper_id, score, signal_breakdown, method_card}`. Adding keys is fine; renaming
  or removing breaks every page.
- **Streamlit page filenames** (`N_<emoji>_<Name>.py`) — Streamlit uses the leading number
  to order sidebar nav and the rest for the label. Don't rename without checking that
  cross-page `st.switch_page(...)` calls also update.
- **`st.session_state["selected_paper_id"]`** is the deep-link contract from Tab 1 to
  Tabs 2 and 3. Other state keys (`_run_match_for`, `_graph_query`) are page-local and you
  may rename freely.

## Known issues / gotchas

- **First page load is slow** (~10-30 s) because `@st.cache_resource` lazily loads
  SPECTER2 + cross-encoder. Subsequent loads in the same Streamlit session are instant.
  Don't accidentally bust the cache (e.g. by changing the `get_*` function signatures).
- **PyVis + cwd**: we pass `cdn_resources='in_line'` and write to `tempfile`. If you ever
  see `lib/` or `tmp_graph.html` appear in the repo root, that's a regression — they're in
  `.gitignore` as belt-and-suspenders but should never be created.
- **Streamlit 1.57.0 pin** (`pyproject.toml`). `st.components.v1.html` is deprecated for
  removal after 2026-06-01 — see CLAUDE.md note. Migration to `st.html` is your task before
  end of project.
- **Tab 4 LLM cost**: cheap on DeepSeek (<$0.02/click) but real. If you ever expose the
  app publicly, add a per-session or per-IP daily cap. Today the gate is "user clicks the
  button" — that's enough for the demo.
- **Tab 3 Opposing fallback**: until B ships SciCite and A wires `s2_contexts`, Opposing
  is mechanism-distance-only — the in-UI banner makes this explicit. Don't strip the
  banner; it's the user's signal that the result is a fallback heuristic.

## Tools to learn

- Streamlit multi-page: https://docs.streamlit.io/develop/concepts/multipage-apps (~15 min)
- Streamlit caching (resource vs data): https://docs.streamlit.io/develop/concepts/architecture/caching (~10 min)
- `st.badge` (1.46+): https://docs.streamlit.io/develop/api-reference/text/st.badge
- PyVis: https://pyvis.readthedocs.io (~10 min, mostly done in spike)

## Commands

```bash
uv run streamlit run ui/app.py             # launch the app (default port 8501)
uv run pytest tests/test_ui_api.py -q      # unit tests for the api surface
```
