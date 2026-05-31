"""One paper, rendered as a Streamlit card. Reusable across all tabs."""

from __future__ import annotations

import streamlit as st

from ui.components.reason_tags import render_reason_tags


def render_paper_card(
    paper: dict | None,
    *,
    paper_id: str | None = None,
    score: float | None = None,
    signal_breakdown: dict | None = None,
    show_actions: bool = True,
    action_prefix: str = "",
) -> None:
    """Render title / year / cites / abstract / reason-tags / deep-link buttons."""
    if paper is None:
        st.warning(f"Paper not in current corpus: {paper_id or '<unknown>'}")
        return

    pid = paper.get("paper_id") or paper_id or "?"
    title = paper.get("title") or "<untitled>"
    year = paper.get("year") or "?"
    cites = paper.get("citation_count") or 0
    abstract = paper.get("abstract")

    with st.container(border=True):
        # Top line: title is the most important; year + cite count + score follow.
        st.markdown(f"### {title}")
        cols = st.columns([1, 1, 2])
        cols[0].markdown(f"**Year:** {year}")
        cols[1].markdown(f"**Cites:** {cites:,}")
        if score is not None:
            cols[2].markdown(f"**Score:** `{score:.3f}`")

        if signal_breakdown is not None:
            render_reason_tags(signal_breakdown)

        if abstract:
            with st.expander("Abstract"):
                st.write(abstract)

        if show_actions:
            a, b, _ = st.columns([1, 1, 4])
            if a.button("📋 Method card", key=f"{action_prefix}mc_{pid}"):
                st.session_state["selected_paper_id"] = pid
                st.switch_page("pages/2_📋_Method_Card.py")
            if b.button("🕸 Show in graph", key=f"{action_prefix}gr_{pid}"):
                st.session_state["selected_paper_id"] = pid
                st.switch_page("pages/3_🕸_Citation_Graph.py")
