"""Unit tests for the OpenAlex adapter's retry logic (no network)."""

import pytest
import requests

from data.sources import openalex


class _Resp:
    def __init__(self, status: int = 200, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_get_retries_on_timeout_then_succeeds(monkeypatch):
    """A transient read timeout retries rather than aborting the whole crawl."""
    monkeypatch.setattr(openalex.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.Timeout("read timed out")
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(openalex.requests, "get", fake_get)
    assert openalex._get("http://x", {}) == {"ok": True}
    assert calls["n"] == 2  # first timed out, second succeeded


def test_get_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(openalex.time, "sleep", lambda _s: None)
    seq = [429, 200]
    monkeypatch.setattr(openalex.requests, "get", lambda url, **kw: _Resp(seq.pop(0), {"ok": True}))
    assert openalex._get("http://x", {}) == {"ok": True}


def test_get_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(openalex.time, "sleep", lambda _s: None)

    def always_reset(url, **kwargs):
        raise requests.ConnectionError("connection reset")

    monkeypatch.setattr(openalex.requests, "get", always_reset)
    with pytest.raises(RuntimeError):
        openalex._get("http://x", {}, max_retries=3)
