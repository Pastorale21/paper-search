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
from ui.style import apply_page_style, callout  # noqa: E402

st.set_page_config(page_title="GNN-RecSys 论文检索", layout="wide")
apply_page_style()
st.title("GNN-RecSys 论文检索系统")

st.markdown(
    """
    这个系统面向 GNN-based recommendation 论文检索。它把语义检索、方法卡机制匹配和
    引文图推理放在同一个工作流里,让演示不只停在“搜到相关论文”,还可以解释每篇论文为什么被召回。
    """
)

callout(
    "当前可报告结论",
    "在扩展 gold set 的 paper-query same subset 上,hybrid nDCG@5 为 0.266,略高于 dense 的 0.253。"
    "standalone method_match 暂低于 dense,所以报告中应表述为机制级信号在融合后带来增益。",
    tone="green",
)

st.subheader("语料与覆盖")
stats = api.corpus_stats()
cols = st.columns(4)
cols[0].metric("语料论文数", f"{stats['papers']:,}")
cols[1].metric("含摘要", f"{stats['papers_with_abstract']:,}")
cols[2].metric("已抽取方法卡", f"{stats['method_cards']:,}")
cols[3].metric(
    "引文图",
    f"{stats['graph_edges']:,} 条边",
    delta=f"{stats['graph_nodes']:,} 个节点",
)

st.divider()

st.subheader("演示路线")
st.markdown(
    """
    1. **搜索**: 用 hybrid 检索,看每条结果旁边的 `dense / bm25 / method_match` 信号标签。
    2. **方法卡**: 选一篇论文,查看 `task / backbone / loss / key_idea`,再运行机制相似匹配。
    3. **引文图**: 展示祖先、跨域同机制和对立方法的路径解释。对立方法目前是机制距离 fallback。
    4. **相关工作**: 用户手动点击后才调用 LLM,生成带 `[N]` 引用的段落并做事实核查。
    """
)

st.caption(
    "状态:demo 已具备完整链路。交付前请运行 eval smoke check,并按 docs/eval_findings_d.md 使用最新结论。"
)
