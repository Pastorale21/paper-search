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
from ui.query_params import get_param, set_params  # noqa: E402
from ui.style import apply_page_style, callout  # noqa: E402

st.set_page_config(page_title="引文图 · GNN-RecSys", layout="wide")
apply_page_style()
st.title("🕸 引文图")
st.caption(
    "选择一个锚点论文,然后运行三种推理查询之一。每条结果都带有路径解释——即系统返回它的“原因”。"
)

papers = api.get_papers_by_id()
options = sorted(papers.values(), key=lambda p: -int(p.get("citation_count") or 0))
labels = [
    f"[{p.get('citation_count', 0):>5} 引用] {p.get('title') or '?'} ({p.get('year') or '?'})"
    for p in options
]
ids = [p["paper_id"] for p in options]

query_pid = get_param("paper_id")
default_pid = query_pid or st.session_state.get("selected_paper_id")
default_index = ids.index(default_pid) if default_pid in ids else 0
choice = st.selectbox(
    "锚点论文",
    options=list(range(len(options))),
    index=default_index,
    format_func=lambda i: labels[i],
)
anchor_pid = ids[choice]
anchor = papers[anchor_pid]
st.session_state["selected_paper_id"] = anchor_pid
set_params(paper_id=anchor_pid)

st.markdown(
    f"**锚点:** {anchor.get('title') or '?'} · {anchor.get('year') or '?'} · `{anchor_pid}`"
)

# Button order per the eval feedback: strongest demo (Ancestors) → mid (Cross-domain) →
# weakest fallback (Opposing) last.
col_a, col_b, col_c = st.columns(3)
ancestors_clicked = col_a.button("🌳 祖先", use_container_width=True)
crossdomain_clicked = col_b.button("🌐 跨域同机制", use_container_width=True)
opposing_clicked = col_c.button("⚔️ 对立方法", use_container_width=True)

if ancestors_clicked:
    st.session_state["_graph_query"] = ("ancestors", anchor_pid)
elif crossdomain_clicked:
    st.session_state["_graph_query"] = ("cross_domain", anchor_pid)
elif opposing_clicked:
    st.session_state["_graph_query"] = ("opposing", anchor_pid)

query_state = st.session_state.get("_graph_query")
if not query_state or query_state[1] != anchor_pid:
    callout(
        "选择一种图推理",
        "建议演示顺序:祖先 → 跨域同机制 → 对立方法。对立方法目前会显示机制距离 fallback 说明。",
        tone="gray",
    )
    st.stop()

query_kind, _ = query_state
top_k = st.slider("显示结果数量", min_value=3, max_value=10, value=5)

with st.spinner("正在基于引文图推理..."):
    if query_kind == "ancestors":
        results = api.find_ancestors(anchor_pid, k=top_k)
        kind_label = "方法学祖先"
    elif query_kind == "cross_domain":
        results = api.find_cross_domain(anchor_pid, k=top_k)
        kind_label = "跨域同机制"
    else:
        # Belt-and-suspenders survey filter on the live mechanism-distance fallback.
        results = api.find_opposing(anchor_pid, k=top_k * 2)
        results = api.filter_survey_titles(results, papers)[:top_k]
        kind_label = "对立方法(机制距离回退)"

st.subheader(kind_label)

if not results:
    callout(
        "当前锚点没有可展示路径",
        "这通常是语料内引用边太稀疏导致的,尤其是经典 foundational papers。请换一篇更新的论文作为锚点。",
        tone="orange",
    )
    st.stop()

if query_kind == "opposing":
    callout(
        "对立方法是 fallback",
        "磁盘上还没有逐边 citation intent,因此这里按 1-hop 邻居上的机制距离排序,并过滤综述类标题。"
        "不要把它讲成真正的“观点反对”。",
        tone="orange",
    )

# Two-column layout: graph on the left, ranked list on the right.
col_g, col_l = st.columns([3, 2])

with col_g:
    paths = [p for r in results for p in r.paths]
    if paths:
        render_graph(anchor_pid, paths, papers, height=520)
    else:
        callout("图视图不可用", "后端返回了候选结果,但没有可绘制的路径。", tone="gray")

with col_l:
    st.markdown("#### 排序结果 + 原因")
    for i, r in enumerate(results, 1):
        paper = papers.get(r.paper_id, {})
        with st.container(border=True):
            st.markdown(f"**{i}. {paper.get('title') or '?'}** · {paper.get('year') or '?'}")
            st.caption(f"`{r.paper_id}` · 分数 `{r.score:.3f}`")
            for p in r.paths[:2]:
                st.markdown(f"› {p.explanation}")
