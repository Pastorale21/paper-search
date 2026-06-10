"""Related Work Draft tab — wires hybrid retrieval + method cards + LLM into a paragraph.

The LLM call is fully wired but only fires when the user clicks **Generate**.
Prompt template lives in ``ui/related_work_prompt.py`` — that's D's iteration surface.
"""

from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui import api  # noqa: E402
from ui.components.reason_tags import render_reason_tags  # noqa: E402
from ui.related_work_prompt import (  # noqa: E402
    build_messages,
    extract_citation_markers,
    parse_llm_response,
    validate_references,
)
from ui.style import apply_page_style, callout, section_label  # noqa: E402

DEMO_DRAFT = (
    "We study graph contrastive learning for collaborative filtering under sparse "
    "user-item interactions. Existing graph recommenders often rely on structural "
    "augmentations or deeper message passing, which can introduce noise and increase "
    "training cost. Our idea is to preserve the lightweight propagation of graph "
    "collaborative filtering while constructing mechanism-aware contrastive views from "
    "user-item neighborhoods and item co-occurrence signals. The goal is to improve "
    "recommendation robustness without depending on expensive graph perturbations."
)

st.set_page_config(page_title="相关工作 · GNN-RecSys", layout="wide")
apply_page_style()
st.title("✍️ 相关工作草稿")
st.caption(
    "粘贴一段草稿想法或摘要 → 系统通过 hybrid 检索召回 top 候选论文,加载它们的"
    "方法卡,并请 LLM 生成一段带 [N] 引用标记的连贯相关工作段落。"
)
if not api.is_llm_configured():
    callout(
        "生成暂未启用",
        "当前未配置 LLM_API_KEY。你仍然可以召回候选论文并检查证据;配置 key 后再生成段落。",
        tone="gray",
    )


def _render_candidate(index: int, item: dict) -> None:
    """Render one retrieved paper as compact evidence for the LLM prompt."""
    paper = item["paper"] or {}
    card = item.get("method_card")
    with st.container(border=True):
        st.markdown(
            f"**[{index}] {paper.get('title') or '?'}** · {paper.get('year') or '?'}  \n"
            f"`{item['paper_id']}` · 分数 `{item['score']:.3f}`"
        )
        render_reason_tags(item.get("signal_breakdown"))
        if card is None:
            st.caption("本地没有方法卡;prompt 将只使用标题和年份。")
            return
        section_label("方法卡证据")
        lines = []
        if card.task:
            lines.append(f"任务: {card.task}")
        if card.backbone:
            lines.append(f"骨干: {card.backbone}")
        if card.loss:
            lines.append(f"损失: {card.loss}")
        if card.key_idea:
            lines.append(f"核心思想: {card.key_idea}")
        st.caption(" · ".join(lines) if lines else "方法卡字段为空。")

load_demo = st.button("加载 demo 摘要")
if load_demo:
    st.session_state["related_work_draft"] = DEMO_DRAFT

user_input = st.text_area(
    "你的草稿想法 / 摘要",
    height=180,
    placeholder="粘贴 1-3 段描述你想法的文字。输入越丰富,召回的候选越好,生成的段落也越有用。",
    key="related_work_draft",
)

cols = st.columns(2)
n_citations = cols[0].slider("使用的引用数量", min_value=5, max_value=15, value=8)
target_words = cols[1].slider("目标段落长度(词数)", min_value=150, max_value=400, value=250)

st.divider()

st.subheader("1. 召回证据")
retrieve = st.button(
    "召回候选论文",
    type="secondary",
    disabled=not user_input.strip(),
)
cached = st.session_state.get("_related_work_retrieval")
same_retrieval = (
    cached
    and cached.get("draft") == user_input.strip()
    and cached.get("n_citations") == n_citations
)
retrieved = cached["results"] if same_retrieval else None
stale_retrieval = bool(cached and not same_retrieval)

st.subheader("2. 生成段落")
generate = st.button(
    "调用 LLM 生成相关工作段落",
    type="primary",
    disabled=not user_input.strip() or not retrieved,
)

if not user_input.strip():
    callout(
        "等待草稿",
        "粘贴你的 idea 或摘要后再生成。LLM 调用只会在你点击按钮后发生。",
        tone="gray",
    )

if retrieve:
    with st.spinner("正在召回候选论文..."):
        retrieved = api.search(user_input.strip(), mode="paper", method="hybrid", k=n_citations)
    st.session_state["_related_work_retrieval"] = {
        "draft": user_input.strip(),
        "n_citations": n_citations,
        "results": retrieved,
    }
    if not retrieved:
        callout(
            "检索返回 0 个候选",
            "请尝试更长、更具体的草稿,尤其补充模型机制、任务和数据场景。",
            tone="orange",
        )
        st.stop()

if retrieved:
    st.subheader(f"已召回 {len(retrieved)} 篇候选论文")
    cards_available = sum(1 for item in retrieved if item.get("method_card") is not None)
    st.caption(f"其中 {cards_available}/{len(retrieved)} 篇带有本地方法卡证据。")
    for i, r in enumerate(retrieved, 1):
        _render_candidate(i, r)
elif user_input.strip():
    if stale_retrieval:
        callout(
            "候选已过期",
            "草稿或引用数量已变化。请重新召回候选论文,避免用旧证据生成段落。",
            tone="orange",
        )
    callout(
        "候选论文尚未召回",
        "先点击“召回候选论文”查看证据列表。确认候选合理后,再点击“调用 LLM 生成相关工作段落”。",
        tone="blue",
    )

if cached:
    if st.button("清除候选缓存"):
        st.session_state.pop("_related_work_retrieval", None)
        st.rerun()

if generate and retrieved:
    messages = build_messages(user_input.strip(), retrieved, target_words=target_words)

    st.subheader("LLM 调用")
    if not api.is_llm_configured():
        callout(
            "LLM_API_KEY 未配置",
            "候选论文证据仍可展示;如需生成段落,请按 nlp/HANDOFF.md 在 .env 中配置 key 后重新加载页面。",
            tone="orange",
        )
        st.stop()

    try:
        client = api.get_llm_client()
    except RuntimeError as e:
        callout("无法初始化 LLM 客户端", str(e), tone="red")
        st.stop()

    with st.spinner(f"正在调用 {api.llm_model_name()} ..."):
        try:
            resp = client.chat.completions.create(
                model=api.llm_model_name(),
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            callout("LLM 调用失败", str(e), tone="red")
            st.stop()

    parsed = parse_llm_response(raw)
    paragraph = parsed.get("paragraph") or ""
    references = parsed.get("references") or []

    st.subheader("生成的段落")
    if parsed.get("_parse_error"):
        callout(
            "LLM 返回了非 JSON 输出",
            "页面已启用解析回退。请在事实核查区查看原始响应,必要时继续迭代 prompt。",
            tone="orange",
        )
    st.markdown(paragraph)

    markers = extract_citation_markers(paragraph)
    if markers:
        st.caption(f"段落中发现的引用标记:{markers}")
    reference_issues = validate_references(markers, references, retrieved)
    if reference_issues:
        callout(
            "引用一致性需要检查",
            "；".join(reference_issues),
            tone="red",
        )

    st.subheader("参考文献")
    paper_by_id = api.get_papers_by_id()
    if not references:
        callout("缺少参考文献列表", "LLM 未返回 references 字段;见下方事实核查。", tone="orange")
    else:
        for ref in references:
            n = ref.get("n")
            pid = ref.get("paper_id") or "?"
            reason = ref.get("one_line_reason") or ""
            paper = paper_by_id.get(pid, {})
            st.markdown(
                f"**[{n}]** {paper.get('title') or '?'} · {paper.get('year') or '?'} · "
                f"`{pid}`  \n_{reason}_"
            )

    with st.expander("🔍 事实核查(原始召回论文 + LLM 响应)"):
        st.markdown("**逐引用来源核查**——每个 [N] 标记都应映射到一篇真实召回的论文:")
        for n, item in enumerate(retrieved, 1):
            paper = item["paper"] or {}
            first_sent = (paper.get("abstract") or "").split(". ")[:1]
            first_sent = first_sent[0] + ("." if first_sent else "")
            st.markdown(
                f"- **[{n}]** `{item['paper_id']}` — {paper.get('title') or '?'}  \n"
                f"  摘要开头:_{first_sent or '(无摘要)'}_"
            )
        st.markdown("**原始 LLM 响应(供 D 迭代 prompt):**")
        st.code(raw, language="json")
        st.markdown("**发送给 LLM 的消息(prompt 输入):**")
        st.code(json.dumps(messages, indent=2, ensure_ascii=False), language="json")
