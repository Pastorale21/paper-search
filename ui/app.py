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
from ui.style import apply_page_style, callout, demo_card  # noqa: E402

st.set_page_config(page_title="GNN-RecSys 论文检索", layout="wide")
apply_page_style()
st.title("GNN-RecSys 论文检索系统")

st.markdown("""
    这个系统面向 GNN-based recommendation 论文检索。它把语义检索、方法卡机制匹配和
    引文图推理放在同一个工作流里,让演示不只停在“搜到相关论文”,还可以解释每篇论文为什么被召回。
    """)

callout(
    "当前可报告结论",
    "在扩展 gold set 的 paper-query same subset 上,hybrid nDCG@5 为 0.266,略高于 dense 的 0.253。"
    "standalone method_match 暂低于 dense,所以报告中应表述为机制级信号在融合后带来增益。",
    tone="green",
)

st.subheader("语料与覆盖")
health = api.cache_health()
missing = [name for name, ok in health.items() if not ok]
if missing:
    callout(
        "本地缓存不完整",
        "缺少: " + "、".join(missing) + "。请先运行 `uv run python -m spike` 重建演示缓存。",
        tone="orange",
    )
    st.stop()

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

coverage_cols = st.columns(2)
card_coverage = stats["method_cards"] / max(stats["papers"], 1)
abstract_coverage = stats["papers_with_abstract"] / max(stats["papers"], 1)
isolated_ratio = stats["graph_isolated"] / max(stats["graph_nodes"], 1)
with coverage_cols[0]:
    st.caption("方法卡覆盖")
    st.progress(card_coverage, text=f"{stats['method_cards']:,}/{stats['papers']:,}")
with coverage_cols[1]:
    st.caption("摘要覆盖 / 图孤立节点")
    st.progress(abstract_coverage, text=f"摘要 {abstract_coverage:.0%}")
    st.progress(1 - isolated_ratio, text=f"非孤立节点 {1 - isolated_ratio:.0%}")

st.divider()

st.subheader("演示路线")
route_cols = st.columns(3)
with route_cols[0]:
    demo_card(
        "1. Hybrid 搜索",
        "加载短查询后运行搜索,重点展示每条结果的 dense / BM25 / method_match 信号标签。",
        code="?query=graph%20contrastive%20learning%20for%20recommendation&mode=short&method=hybrid&k=10",
    )
with route_cols[1]:
    demo_card(
        "2. 方法卡与图推理",
        "从搜索结果跳到方法卡,再运行机制相似匹配;或切到引文图展示路径解释。",
        code="?paper_id=W3004578093",
    )
with route_cols[2]:
    demo_card(
        "3. 相关工作草稿",
        "先召回候选证据,确认论文和方法卡合理后,再手动调用 LLM 生成带 [N] 标记的段落。",
        code="Tab 4: 召回候选论文 → 调用 LLM",
    )

st.caption(
    "状态:demo 已具备完整链路。"
    "交付前请运行 eval smoke check,并按 docs/eval_findings_d.md 使用最新结论。"
)
