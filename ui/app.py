"""Landing page for the 4-tab GNN-recsys paper-search UI.

Streamlit auto-discovers ``ui/pages/*.py`` and renders the sidebar nav. This file is just
the welcome screen — every real interaction lives in a page module.
"""

from __future__ import annotations

import pathlib
import sys

# Make project root importable so ``streamlit run ui/app.py`` resolves package imports.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import api  # noqa: E402

st.set_page_config(page_title="GNN-RecSys Paper Search", layout="wide")
st.title("GNN-RecSys Paper Search")

st.markdown("""
    A research-paper search system whose differentiation is **mechanism-level matching**
    plus **multi-hop citation-graph reasoning**. Standard dense retrieval saturates inside
    topic clusters with a top-5 cosine spread of ~0.006 — useless for picking the *right*
    GNN-recsys paper. This system breaks ties by comparing **method cards**
    (`task / backbone / loss / key_idea`) and by walking the citation graph along
    intent-weighted paths. Eval against a 10-query gold set (same-subset, paper queries):
    **method_match +0.112 nDCG@5 vs dense**; **hybrid +0.091 vs dense**.
    """)

st.subheader("Corpus & coverage")
stats = api.corpus_stats()
cols = st.columns(4)
cols[0].metric("Papers in corpus", f"{stats['papers']:,}")
cols[1].metric("With abstracts", f"{stats['papers_with_abstract']:,}")
cols[2].metric("Method cards extracted", f"{stats['method_cards']:,}")
cols[3].metric(
    "Citation graph",
    f"{stats['graph_edges']:,} edges",
    delta=f"{stats['graph_nodes']:,} nodes",
)

st.divider()

st.subheader("How to use this app")
st.markdown("""
    The sidebar has four tabs:

    1. **🔍 Search** — semantic + hybrid retrieval. Each result carries *reason tags*
       (`dense / bm25 / method_match`) so you can see which signal surfaced it.
    2. **📋 Method Card** — structured `task / backbone / loss / key_idea` per paper, plus a
       **"Find similar mechanism"** button that shows **per-field cosines** (`backbone: 0.91,
       loss: 0.95, key_idea: 0.78`) — the visible evidence of mechanism-level matching.
    3. **🕸 Citation Graph** — pick a paper, run one of three reasoning queries
       (**Ancestors / Cross-domain / Opposing**), and get an interactive subgraph plus
       human-readable *path explanations*.
    4. **✍️ Related Work Draft** — paste your own idea or abstract, generate a related-work
       paragraph backed by retrieved papers and their method cards.
    """)

st.caption(
    "Status: scaffolded for demo. Visual polish, prompt iteration, and link-sharing are "
    "module owner D's next tasks — see ``ui/HANDOFF.md``."
)
