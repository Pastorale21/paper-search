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
from ui.query_params import get_int_param, get_param, set_params  # noqa: E402

st.set_page_config(page_title="搜索 · GNN-RecSys", layout="wide")
st.title("🔍 搜索")
st.caption("短查询模式(short)用于自然语言查询;论文模式(paper)用于以论文为查询(粘贴一段草稿摘要)。")

initial_mode = get_param("mode", "short")
initial_method = get_param("method", "hybrid")
initial_query = get_param("query", "") or ""
initial_k = get_int_param("k", 10, allowed=range(3, 21))
method_options = ["hybrid", "dense", "bm25", "dense_rerank"]

mode_label = st.radio(
    "查询模式",
    ["短查询 (short)", "粘贴论文摘要 (paper)"],
    index=1 if initial_mode == "paper" else 0,
    horizontal=True,
    label_visibility="visible",
)
mode = "paper" if mode_label.startswith("粘贴") else "short"

cols = st.columns([1, 1])
method = cols[0].selectbox(
    "检索方法",
    options=method_options,
    index=method_options.index(initial_method) if initial_method in method_options else 0,
    help="Hybrid 通过 RRF 融合 dense + BM25 + method_match。"
    "只有 hybrid 会为每条结果生成检索信号标签。",
)
top_k = cols[1].slider("返回数量 (Top-K)", min_value=3, max_value=20, value=initial_k)

placeholder = (
    "如:Collaborative filtering benefits from graph convolution..."
    if mode == "paper"
    else "如:graph contrastive learning for recommendation"
)
if mode == "paper":
    query = st.text_area("查询", height=140, placeholder=placeholder, value=initial_query)
else:
    query = st.text_input("查询", placeholder=placeholder, value=initial_query)

if st.button("搜索", type="primary", disabled=not query.strip()):
    set_params(query=query.strip(), mode=mode, method=method, k=top_k)
    with st.spinner("检索中..."):
        results = api.search(query.strip(), mode=mode, method=method, k=top_k)
    if not results:
        st.info("无结果。")
    else:
        st.subheader(f"`{method}` 的 Top-{len(results)} 结果")
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
