"""Citation Graph tab — three reasoning queries rendered through PyVis + path explanations."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import api  # noqa: E402
from ui.components.graph_view import render_graph  # noqa: E402

st.set_page_config(page_title="Citation Graph · GNN-RecSys", layout="wide")
st.title("🕸 Citation Graph")
st.caption(
    "Pick an anchor, then run one of three reasoning queries. Each result carries a "
    "path explanation — the 'why' the system returned it."
)

papers = api.get_papers_by_id()
options = sorted(papers.values(), key=lambda p: -int(p.get("citation_count") or 0))
labels = [
    f"[{p.get('citation_count', 0):>5} cites] {p.get('title') or '?'} ({p.get('year') or '?'})"
    for p in options
]
ids = [p["paper_id"] for p in options]

default_pid = st.session_state.get("selected_paper_id")
default_index = ids.index(default_pid) if default_pid in ids else 0
choice = st.selectbox(
    "Anchor paper",
    options=list(range(len(options))),
    index=default_index,
    format_func=lambda i: labels[i],
)
anchor_pid = ids[choice]
anchor = papers[anchor_pid]
st.session_state["selected_paper_id"] = anchor_pid

st.markdown(
    f"**Anchor:** {anchor.get('title') or '?'} · {anchor.get('year') or '?'} · `{anchor_pid}`"
)

# Button order per the eval feedback: strongest demo (Ancestors) → mid (Cross-domain) →
# weakest fallback (Opposing) last.
col_a, col_b, col_c = st.columns(3)
ancestors_clicked = col_a.button("🌳 Ancestors", use_container_width=True)
crossdomain_clicked = col_b.button("🌐 Cross-domain same mechanism", use_container_width=True)
opposing_clicked = col_c.button("⚔️ Opposing", use_container_width=True)

if ancestors_clicked:
    st.session_state["_graph_query"] = ("ancestors", anchor_pid)
elif crossdomain_clicked:
    st.session_state["_graph_query"] = ("cross_domain", anchor_pid)
elif opposing_clicked:
    st.session_state["_graph_query"] = ("opposing", anchor_pid)

query_state = st.session_state.get("_graph_query")
if not query_state or query_state[1] != anchor_pid:
    st.info("Pick a query button above to see the reasoning trace.")
    st.stop()

query_kind, _ = query_state
top_k = st.slider("How many results to display", min_value=3, max_value=10, value=5)

with st.spinner("Reasoning over the citation graph..."):
    if query_kind == "ancestors":
        results = api.find_ancestors(anchor_pid, k=top_k)
        kind_label = "Methodological ancestors"
    elif query_kind == "cross_domain":
        results = api.find_cross_domain(anchor_pid, k=top_k)
        kind_label = "Cross-domain same mechanism"
    else:
        # Belt-and-suspenders survey filter on the live mechanism-distance fallback.
        results = api.find_opposing(anchor_pid, k=top_k * 2)
        results = api.filter_survey_titles(results, papers)[:top_k]
        kind_label = "Opposing (mechanism-distance fallback)"

st.subheader(kind_label)

if not results:
    st.warning(
        "No results — see `retrieval/HANDOFF.md` (graph reasoning, OUT-sparsity). "
        "Try a more recent paper as the anchor."
    )
    st.stop()

if query_kind == "opposing":
    st.caption(
        "ℹ️ Opposing currently runs on the **mechanism-distance fallback** — no per-edge "
        "intent metadata is on disk yet. Each result is ranked by `1 - similarity` to the "
        "anchor over 1-hop neighbors (cites ∪ cited-by); survey-titled papers are filtered "
        "on top of the empty-card exclusion. The explanation banner makes the fallback "
        "visible."
    )

# Two-column layout: graph on the left, ranked list on the right.
col_g, col_l = st.columns([3, 2])

with col_g:
    paths = [p for r in results for p in r.paths]
    if paths:
        render_graph(anchor_pid, paths, papers, height=520)
    else:
        st.info("Graph view unavailable (no paths returned).")

with col_l:
    st.markdown("#### Ranked results + why")
    for i, r in enumerate(results, 1):
        paper = papers.get(r.paper_id, {})
        with st.container(border=True):
            st.markdown(f"**{i}. {paper.get('title') or '?'}** · {paper.get('year') or '?'}")
            st.caption(f"`{r.paper_id}` · score `{r.score:.3f}`")
            for p in r.paths[:2]:
                st.markdown(f"› {p.explanation}")
