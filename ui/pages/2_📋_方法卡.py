"""Method Card tab — structured view + Find Similar Mechanism (the showcase)."""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import api  # noqa: E402
from ui.query_params import get_param, selected_paper_link_hint, set_params  # noqa: E402
from ui.style import apply_page_style, callout, section_label  # noqa: E402

st.set_page_config(page_title="方法卡 · GNN-RecSys", layout="wide")
apply_page_style()
st.title("📋 方法卡")

# Chinese display labels for method-card fields; English values are preserved as-is.
_FIELD_LABELS = {
    "task": "任务",
    "input": "输入",
    "output": "输出",
    "backbone": "骨干网络",
    "loss": "损失",
    "key_idea": "核心思想",
}

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
    "选择论文",
    options=list(range(len(options))),
    index=default_index,
    format_func=lambda i: labels[i],
)
selected_pid = ids[choice]
selected = papers[selected_pid]
st.session_state["selected_paper_id"] = selected_pid
set_params(paper_id=selected_pid)

card = api.load_method_card(selected_pid)

# --- Top: structured card + abstract -----------------------------------------------------

st.markdown(f"### {selected.get('title') or '?'} · {selected.get('year') or '?'}")
st.caption(f"`{selected_pid}` · {selected.get('citation_count', 0):,} 次引用")
st.caption(f"深链参数: `{selected_paper_link_hint(selected_pid)}`")

col_card, col_abs = st.columns([3, 2])

with col_card:
    if card is None:
        callout(
            "该论文尚未抽取方法卡",
            "按照团队约定,UI 不会自动触发付费 LLM 抽取。需要先运行下面的 CLI 生成缓存。",
            tone="orange",
        )
        st.code(
            "uv run python -m nlp.method_card.extractor --top 400",
            language="bash",
        )
        st.caption("可以先用更小的 `--top N` 试跑;成本估算见 `nlp/HANDOFF.md`。")
    else:
        section_label("机制级字段")
        missing_core = [
            label
            for field, label in (
                ("task", "任务"),
                ("backbone", "骨干网络"),
                ("loss", "损失"),
                ("key_idea", "核心思想"),
            )
            if not getattr(card, field)
        ]
        if missing_core:
            callout(
                "方法卡字段不完整",
                "缺失字段:" + "、".join(missing_core) + "。这会降低 standalone method_match 的稳定性。",
                tone="orange",
            )
        st.markdown(
            f"**🎯 任务:** {card.task or '_(空)_'}  \n"
            f"**📥 输入:** {card.input or '_(空)_'}  \n"
            f"**📤 输出:** {card.output or '_(空)_'}  \n"
            f"**🏗 骨干网络:** {card.backbone or '_(空)_'}  \n"
            f"**📐 损失:** {card.loss or '_(空)_'}  \n"
            f"**💡 核心思想:** *{card.key_idea or '_(空)_'}*"
        )
        if card.datasets:
            st.markdown("**🗂 数据集:** " + " ".join(f":blue-badge[{d}]" for d in card.datasets))
        if card.metrics:
            st.markdown("**📊 指标:** " + " ".join(f":violet-badge[{m}]" for m in card.metrics))

with col_abs:
    section_label("摘要")
    abstract = selected.get("abstract") or "_(本地无摘要)_"
    st.write(abstract)

st.divider()

# --- The showcase: Find Similar Mechanism (prominent, not buried) ------------------------

st.header("🔍 查找机制相似的论文")
callout(
    "可见证据",
    "每个候选都会展示逐字段余弦相似度。字段权重用于聚合打分:"
    f" {api.field_weights()}。",
    tone="blue",
)

if card is None:
    callout("无法运行机制匹配", "运行相似度排序前需要锚点论文的方法卡。", tone="gray")
else:
    if st.button(
        "在全语料上运行机制匹配",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["_run_match_for"] = selected_pid

if st.session_state.get("_run_match_for") == selected_pid and card is not None:
    with st.spinner("正在按加权字段余弦为全部语料论文打分..."):
        matches = api.match_similar_mechanism(selected_pid, k=10)
    if not matches:
        callout(
            "没有可展示的机制匹配",
            "锚点论文的方法卡可能为空,或候选论文缺少可比较字段。",
            tone="orange",
        )
    else:
        st.markdown(f"### Top-{len(matches)} 机制匹配论文")
        weights = api.field_weights()
        for i, m in enumerate(matches, 1):
            with st.container(border=True):
                paper = m["paper"] or {}
                pid = m["paper_id"]
                title = paper.get("title") or "?"
                yr = paper.get("year") or "?"
                st.markdown(f"**{i}. {title}** · {yr}  \n`{pid}` · 加权分数 **`{m['score']:.3f}`**")

                # Per-field cosines as colored chips. This IS the visible evidence.
                pf = m["per_field"]
                chip_row = []
                for f, cos in pf.items():
                    w = weights.get(f, 0.0)
                    label = _FIELD_LABELS.get(f, f)
                    if cos is None:
                        chip_row.append(f":gray-badge[{label} (w={w:.2f}):无数据]")
                    else:
                        color = "green" if cos >= 0.9 else "blue" if cos >= 0.75 else "orange"
                        chip_row.append(f":{color}-badge[{label} (w={w:.2f}): {cos:.3f}]")
                st.markdown(" ".join(chip_row))

                cand_card = m["method_card"]
                if cand_card is not None and cand_card.backbone:
                    st.caption(f"**骨干网络:** {cand_card.backbone} · **损失:** {cand_card.loss}")
                with st.expander("摘要"):
                    st.write(paper.get("abstract") or "_(本地无摘要)_")
