"""Shared Streamlit styling and small presentation helpers."""

from __future__ import annotations

import html

import streamlit as st


_CSS = """
<style>
:root {
  --paper-border: #d8dee9;
  --paper-muted: #5f6b7a;
  --paper-soft: #f7f9fc;
  --paper-blue: #2563eb;
}
.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
}
[data-testid="stMetric"] {
  background: var(--paper-soft);
  border: 1px solid var(--paper-border);
  border-radius: 8px;
  padding: 0.8rem 1rem;
}
.paper-callout {
  border: 1px solid var(--paper-border);
  border-left: 4px solid var(--paper-blue);
  border-radius: 8px;
  background: #ffffff;
  padding: 0.85rem 1rem;
  margin: 0.75rem 0 1rem;
}
.paper-callout-title {
  font-weight: 700;
  margin-bottom: 0.25rem;
}
.paper-callout-body {
  color: var(--paper-muted);
  line-height: 1.55;
}
.paper-meta {
  color: var(--paper-muted);
  font-size: 0.92rem;
}
.paper-section-label {
  color: var(--paper-muted);
  font-size: 0.86rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}
.paper-mini-card {
  border: 1px solid var(--paper-border);
  border-radius: 8px;
  background: #ffffff;
  padding: 0.85rem 1rem;
  margin: 0.45rem 0;
}
.paper-mini-title {
  font-weight: 700;
  line-height: 1.35;
  margin-bottom: 0.25rem;
}
.paper-mini-body {
  color: var(--paper-muted);
  font-size: 0.92rem;
  line-height: 1.5;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
  white-space: nowrap;
}
</style>
"""


def apply_page_style() -> None:
    """Install shared CSS for the Streamlit UI."""
    st.markdown(_CSS, unsafe_allow_html=True)


def callout(title: str, body: str, *, tone: str = "blue") -> None:
    """Render a compact explanatory status block."""
    colors = {
        "blue": "#2563eb",
        "green": "#16a34a",
        "orange": "#ea580c",
        "red": "#dc2626",
        "gray": "#6b7280",
    }
    border = colors.get(tone, colors["blue"])
    safe_title = html.escape(title)
    safe_body = html.escape(body)
    st.markdown(
        f"""
        <div class="paper-callout" style="border-left-color:{border}">
          <div class="paper-callout-title">{safe_title}</div>
          <div class="paper-callout-body">{safe_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def meta(text: str) -> None:
    """Render muted one-line metadata."""
    st.markdown(f'<div class="paper-meta">{html.escape(text)}</div>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    """Render a compact all-caps section label."""
    st.markdown(
        f'<div class="paper-section-label">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )
