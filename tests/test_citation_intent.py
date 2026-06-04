"""Unit tests for citation intent classification (no real LLM API)."""

from unittest.mock import MagicMock

import pytest

from nlp.citation_intent import classifier
from nlp.citation_intent.classifier import (
    CitationIntentClassifier,
    classify_with_scicite,
    map_s2_intent,
    map_scicite_label,
)


def test_s2_intent_mapping():
    assert map_s2_intent("methodology") == "method"
    assert map_s2_intent("background") == "background"
    assert map_s2_intent("result") == "comparison"
    assert map_s2_intent("Methodology") == "method"  # case-insensitive
    assert map_s2_intent("unknown-label") == "background"  # safe default


def test_scicite_label_mapping():
    assert map_scicite_label("background") == "background"
    assert map_scicite_label("method") == "method"
    assert map_scicite_label("result") == "comparison"
    assert map_scicite_label("LABEL_2") == "comparison"


def test_classify_uses_s2_intents_when_present():
    clf = CitationIntentClassifier(use_scicite=False)
    # dict carrying S2 intents -> mapped without any LLM call
    assert clf.classify({"intents": ["methodology"], "context": "..."}) == "method"
    assert clf.classify({"intents": ["result"]}) == "comparison"


def test_classify_dispatches_to_scicite_when_flag_set(monkeypatch):
    def fake_pipeline(text, truncation=True):
        return [{"label": "method", "score": 0.9}]

    classifier._get_scicite_pipeline.cache_clear()
    monkeypatch.setattr(classifier, "_get_scicite_pipeline", lambda: fake_pipeline)
    clf = CitationIntentClassifier(use_scicite=True)
    assert clf.classify("We adopt the loss from [CITATION].") == "method"


def test_classify_with_scicite_empty_context_defaults_background():
    assert classify_with_scicite("") == "background"


def test_scicite_pipeline_missing_model_raises(tmp_path, monkeypatch):
    classifier._get_scicite_pipeline.cache_clear()
    monkeypatch.setattr(classifier.config, "SCICITE_MODEL_DIR", tmp_path / "missing_scicite")

    with pytest.raises(FileNotFoundError, match="Train it with"):
        classifier._get_scicite_pipeline()
    classifier._get_scicite_pipeline.cache_clear()


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
