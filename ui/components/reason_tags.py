"""Render a hybrid ``signal_breakdown`` as colored chips.

Visible payoff of /scaffold-retrieval — Tab 1 surfaces each signal that surfaced the paper.
"""

from __future__ import annotations

import streamlit as st

# Streamlit's native st.badge accepts: blue, green, red, violet, orange, gray, rainbow, primary.
_SIGNAL_COLORS = {
    "dense": "blue",
    "bm25": "gray",
    "method_match": "green",
    "cross_encoder": "violet",
}


def render_reason_tags(signal_breakdown: dict | None) -> None:
    """Render one chip per signal that surfaced the paper; skip ``None`` values."""
    if not signal_breakdown:
        st.caption("_该方法无逐信号分解_")
        return
    chips: list[str] = []
    skipped: list[str] = []
    for signal, score in signal_breakdown.items():
        color = _SIGNAL_COLORS.get(signal, "primary")
        if score is None:
            skipped.append(signal)
            continue
        chips.append(f":{color}-badge[{signal}: {score:.3f}]")
    if chips:
        st.caption("检索信号(哪些信号召回了该论文)")
        st.markdown(" ".join(chips))
    if skipped:
        st.caption("未召回该论文的信号:" + ", ".join(skipped))
