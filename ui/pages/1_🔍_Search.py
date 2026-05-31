"""Search tab — semantic + hybrid retrieval with reason-tag chips."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import api  # noqa: E402
from ui.components.paper_card import render_paper_card  # noqa: E402

st.set_page_config(page_title="Search · GNN-RecSys", layout="wide")
st.title("🔍 Search")
st.caption(
    "Mode = short for natural-language queries; mode = paper for paper-as-query "
    "(paste a draft abstract)."
)

mode_label = st.radio(
    "Query mode",
    ["短查询 (short)", "粘贴论文摘要 (paper)"],
    horizontal=True,
    label_visibility="visible",
)
mode = "paper" if mode_label.startswith("粘贴") else "short"

cols = st.columns([1, 1])
method = cols[0].selectbox(
    "Method",
    options=["hybrid", "dense", "bm25", "dense_rerank"],
    index=0,
    help="Hybrid fuses dense + BM25 + method_match via RRF. "
    "Only hybrid produces per-result reason tags.",
)
top_k = cols[1].slider("Top-K", min_value=3, max_value=20, value=10)

placeholder = (
    "如:Collaborative filtering benefits from graph convolution..."
    if mode == "paper"
    else "如:graph contrastive learning for recommendation"
)
if mode == "paper":
    query = st.text_area("Query", height=140, placeholder=placeholder)
else:
    query = st.text_input("Query", placeholder=placeholder)

if st.button("Search", type="primary", disabled=not query.strip()):
    with st.spinner("Retrieving..."):
        results = api.search(query.strip(), mode=mode, method=method, k=top_k)
    if not results:
        st.info("No results.")
    else:
        st.subheader(f"Top-{len(results)} for `{method}`")
        for i, r in enumerate(results, 1):
            st.markdown(f"#### {i}.")
            render_paper_card(
                r["paper"],
                paper_id=r["paper_id"],
                score=r["score"],
                signal_breakdown=r["signal_breakdown"],
                show_actions=True,
                action_prefix=f"search_{i}_",
            )
