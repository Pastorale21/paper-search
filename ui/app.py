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

st.set_page_config(page_title="GNN-RecSys 论文检索", layout="wide")
st.title("GNN-RecSys 论文检索系统")

st.markdown("""
    一个论文检索系统,核心差异化在于**机制级匹配**与**引文图多跳推理**。标准稠密检索
    在主题簇内部会饱和——top-5 余弦相似度的跨度仅约 0.006,根本无法挑出*正确的*
    GNN 推荐论文。本系统通过比对**方法卡**(`task / backbone / loss / key_idea`)、
    并沿意图加权路径游走引文图来打破这种平局。在 10 条查询的 gold set 上评测(同子域、
    paper 查询):**method_match 相比 dense +0.112 nDCG@5**;**hybrid 相比 dense +0.091**。
    """)

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

st.subheader("使用说明")
st.markdown("""
    侧边栏有四个标签页:

    1. **🔍 搜索** — 语义 + 混合检索。每条结果都带有*检索信号标签*
       (`dense / bm25 / method_match`),便于你看清是哪个信号召回了它。
    2. **📋 方法卡** — 每篇论文的结构化 `task / backbone / loss / key_idea`,外加一个
       **“查找相似机制”**按钮,展示**逐字段余弦相似度**(`backbone: 0.91,
       loss: 0.95, key_idea: 0.78`)——机制级匹配的可见证据。
    3. **🕸 引文图** — 选一篇论文,运行三种推理查询之一
       (**祖先 / 跨域同机制 / 对立方法**),得到一张交互式子图以及
       人类可读的*路径解释*。
    4. **✍️ 相关工作草稿** — 粘贴你自己的想法或摘要,基于召回论文及其方法卡
       生成一段相关工作。
    """)

st.caption(
    "状态:demo 脚手架已搭好。视觉打磨、prompt 迭代与链接分享是模块负责人 D 的"
    "后续任务——见 ``ui/HANDOFF.md``。"
)
