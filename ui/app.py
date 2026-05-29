"""Streamlit + PyVis UI for the GNN-recsys spike: semantic search + citation subgraph."""

from __future__ import annotations

import pathlib
import sys
import tempfile

# Ensure the project root is importable when launched via `streamlit run ui/app.py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from pyvis.network import Network  # noqa: E402

from spike import search  # noqa: E402


@st.cache_resource
def _warm() -> bool:
    """Load index/graph/papers once per session."""
    search._ensure_loaded()
    return True


def _render_graph(paper_ids: list[str], hops: int = 1) -> None:
    g = search.subgraph_for(paper_ids, hops=hops)
    net = Network(height="500px", width="100%", directed=True, notebook=False)
    hits = set(paper_ids)
    for node, data in g.nodes(data=True):
        title = data.get("title") or node
        label = (title[:40] + "…") if len(title) > 40 else title
        net.add_node(
            node,
            label=label,
            title=f"{title} ({data.get('year')})",
            color="#e8543f" if node in hits else "#5b8def",
        )
    for u, v in g.edges():
        net.add_edge(u, v)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        net.write_html(f.name)
        html = pathlib.Path(f.name).read_text()
    components.html(html, height=520)


st.set_page_config(page_title="GNN-recsys Paper Search", layout="wide")
st.title("GNN-recsys 论文语义搜索 (spike)")

_warm()

mode_label = st.radio("查询模式", ["短查询", "粘贴论文摘要"], horizontal=True)
mode = "paper" if mode_label == "粘贴论文摘要" else "short"
top_k = st.slider("Top-K", 3, 20, 10)
placeholder = (
    "粘贴一段论文摘要做 paper-to-paper 检索…"
    if mode == "paper"
    else "如:graph contrastive learning for recommendation"
)
text = st.text_area("查询内容", height=120, placeholder=placeholder)

if st.button("搜索", type="primary") and text.strip():
    results = search.search(text.strip(), mode=mode, top_k=top_k)
    st.subheader(f"命中 {len(results)} 篇")
    for rank, (paper, score) in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(
                f"**{rank}. {paper.title}** · {paper.year} · "
                f"score {score:.3f} · cites {paper.citation_count}"
            )
            if paper.abstract:
                with st.expander("摘要"):
                    st.write(paper.abstract)
    st.subheader("引文子图(红=命中,蓝=1-hop 邻居)")
    _render_graph([p.paper_id for p, _ in results], hops=1)
