"""Unit tests for citation intent classification (no real LLM API)."""

from unittest.mock import MagicMock

import pytest

from nlp.citation_intent.classifier import CitationIntentClassifier, map_s2_intent


def test_s2_intent_mapping():
    assert map_s2_intent("methodology") == "method"
    assert map_s2_intent("background") == "background"
    assert map_s2_intent("result") == "comparison"
    assert map_s2_intent("Methodology") == "method"  # case-insensitive
    assert map_s2_intent("unknown-label") == "background"  # safe default


def test_classify_uses_s2_intents_when_present():
    clf = CitationIntentClassifier(use_scicite=False)
    # dict carrying S2 intents -> mapped without any LLM call
    assert clf.classify({"intents": ["methodology"], "context": "..."}) == "method"
    assert clf.classify({"intents": ["result"]}) == "comparison"


def test_classify_dispatches_to_scicite_when_flag_set():
    clf = CitationIntentClassifier(use_scicite=True)
    with pytest.raises(NotImplementedError):
        clf.classify("any context text")


def test_classify_llm_fallback():
    clf = CitationIntentClassifier(use_scicite=False)
    msg = MagicMock()
    msg.choices = [MagicMock(message=MagicMock(content="comparison"))]
    client = MagicMock()
    client.chat.completions.create.return_value = msg
    clf._client = client  # inject; no real key needed
    assert clf.classify("We outperform [CITATION] on all benchmarks.") == "comparison"


def test_classify_empty_context_defaults_background():
    clf = CitationIntentClassifier(use_scicite=False)
    # empty text short-circuits to 'background' without touching the client
    assert clf.classify("") == "background"
