"""Small helpers for Streamlit query-parameter deep links."""

from __future__ import annotations

from typing import Iterable

import streamlit as st


def get_param(name: str, default: str | None = None) -> str | None:
    """Return one query-param value, handling Streamlit API shape differences."""
    value = st.query_params.get(name)
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return str(value)


def get_int_param(name: str, default: int, *, allowed: Iterable[int] | None = None) -> int:
    """Return an integer query param, falling back when invalid or disallowed."""
    raw = get_param(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        return default
    if allowed is not None and value not in set(allowed):
        return default
    return value


def set_params(**params: str | int | None) -> None:
    """Set or remove query params without disturbing unrelated params."""
    for key, value in params.items():
        if value is None:
            st.query_params.pop(key, None)
        else:
            st.query_params[key] = str(value)


def selected_paper_link_hint(paper_id: str) -> str:
    """Return a short copyable query string for linking to the selected paper."""
    return f"?paper_id={paper_id}"
