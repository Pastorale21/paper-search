"""Shared Streamlit styling and small presentation helpers.

Design tokens: warm "paper" neutrals + a single deep teal-ink accent (matching
`.streamlit/config.toml`). All colors are OKLCH and every neutral is tinted toward the warm hue,
so there is no pure black or white. The public helpers (`apply_page_style`, `callout`, `meta`,
`section_label`, `demo_card`) keep their signatures — only the presentation changed.
"""

from __future__ import annotations

import html

import streamlit as st

# tone -> accent color (OKLCH). "blue" maps to the teal app accent so info callouts stay cohesive.
_TONE_COLORS = {
    "blue": "oklch(0.50 0.075 195)",  # info = app accent (teal ink)
    "green": "oklch(0.53 0.10 150)",  # success / ready
    "orange": "oklch(0.64 0.13 65)",  # warning
    "red": "oklch(0.55 0.17 25)",  # error
    "gray": "oklch(0.55 0.012 70)",  # neutral / waiting
}

_CSS = """
<style>
:root {
  --ink: oklch(0.27 0.012 70);
  --ink-soft: oklch(0.50 0.016 70);
  --surface: oklch(0.992 0.004 85);
  --panel: oklch(0.965 0.006 80);
  --border: oklch(0.90 0.008 80);
  --accent: oklch(0.50 0.075 195);
  --radius: 0.6rem;
}

.block-container { padding-top: 2.4rem; padding-bottom: 3.5rem; }

/* Headings: tighter tracking + clear weight hierarchy */
.block-container h1 { font-weight: 680; letter-spacing: -0.015em; }
.block-container h2 { font-weight: 660; letter-spacing: -0.008em; }
.block-container h3 { font-weight: 650; letter-spacing: -0.005em; }
.block-container h4 { font-weight: 640; line-height: 1.3; }

/* Metric: warm panel, soft border, consistent radius */
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.7rem 1rem;
}

/* Bordered containers (paper cards) get the shared radius */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }

/* Callout: full border + tone-tinted ground + a leading tone dot. No side-stripe border. */
.paper-callout {
  border: 1px solid var(--border);
  border: 1px solid color-mix(in oklch, var(--tone, var(--accent)) 24%, var(--border));
  background: var(--surface);
  background: color-mix(in oklch, var(--tone, var(--accent)) 6%, var(--surface));
  border-radius: var(--radius);
  padding: 0.8rem 1rem;
  margin: 0.7rem 0 1rem;
}
.paper-callout-title {
  display: flex; align-items: center; gap: 0.5rem;
  font-weight: 660; color: var(--ink); margin-bottom: 0.2rem;
}
.paper-dot {
  width: 0.5rem; height: 0.5rem; border-radius: 50%; flex: none;
  background: var(--tone, var(--accent));
}
.paper-callout-body { color: var(--ink-soft); line-height: 1.55; }

.paper-meta { color: var(--ink-soft); font-size: 0.9rem; line-height: 1.5; }

.paper-section-label {
  color: var(--ink-soft);
  font-size: 0.74rem; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
  margin: 0.2rem 0 0.45rem;
}

.paper-mini-card {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: var(--radius);
  padding: 0.85rem 1rem;
  margin: 0.45rem 0;
}
.paper-mini-title { font-weight: 660; line-height: 1.35; margin-bottom: 0.2rem; color: var(--ink); }
.paper-mini-body { color: var(--ink-soft); font-size: 0.9rem; line-height: 1.5; }
.paper-demo-code {
  margin-top: 0.55rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82rem; color: var(--ink);
  background: var(--panel);
  background: color-mix(in oklch, var(--accent) 8%, var(--surface));
  border: 1px solid color-mix(in oklch, var(--accent) 16%, var(--border));
  border-radius: 0.45rem;
  padding: 0.45rem 0.6rem;
  word-break: break-word;
}

div[data-testid="stHorizontalBlock"] button[kind="secondary"] { white-space: nowrap; }
</style>
"""


def apply_page_style() -> None:
    """Install shared CSS for the Streamlit UI."""
    st.markdown(_CSS, unsafe_allow_html=True)


def callout(title: str, body: str, *, tone: str = "blue") -> None:
    """Render a compact explanatory status block (full border + tone dot, no side-stripe)."""
    color = _TONE_COLORS.get(tone, _TONE_COLORS["blue"])
    st.markdown(
        f"""
        <div class="paper-callout" style="--tone:{color}">
          <div class="paper-callout-title"><span class="paper-dot"></span>{html.escape(title)}</div>
          <div class="paper-callout-body">{html.escape(body)}</div>
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


def demo_card(title: str, body: str, *, code: str | None = None) -> None:
    """Render a compact demo route card."""
    safe_code = f'<div class="paper-demo-code">{html.escape(code)}</div>' if code else ""
    st.markdown(
        f"""
        <div class="paper-mini-card">
          <div class="paper-mini-title">{html.escape(title)}</div>
          <div class="paper-mini-body">{html.escape(body)}</div>
          {safe_code}
        </div>
        """,
        unsafe_allow_html=True,
    )
