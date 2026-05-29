"""Unit tests for method card prompts + extractor (no real LLM API)."""

from unittest.mock import MagicMock

import pytest

from nlp.method_card.extractor import MethodCardExtractor, _parse_json
from nlp.method_card.prompts import build_prompt
from schemas import Paper


def test_prompt_includes_few_shots():
    msgs = build_prompt(abstract="An abstract.", title="A Title")
    # system + 3 (user, assistant) few-shot pairs + 1 final user = 8 messages.
    assert len(msgs) >= 7
    assert msgs[0]["role"] == "system"
    assert "JSON" in msgs[0]["content"]  # DeepSeek json_object mode requires the word
    assert msgs[-1]["role"] == "user"
    assert "A Title" in msgs[-1]["content"]


def test_robust_json_parsing():
    payload = {"task": "rec", "datasets": ["Gowalla"]}
    assert _parse_json('{"task": "rec", "datasets": ["Gowalla"]}') == payload
    assert _parse_json('```json\n{"task": "rec", "datasets": ["Gowalla"]}\n```') == payload
    assert _parse_json('Here is the JSON: {"task": "rec", "datasets": ["Gowalla"]}') == payload


def test_parse_json_raises_on_garbage():
    with pytest.raises(ValueError):  # json.JSONDecodeError subclasses ValueError
        _parse_json("no json object here at all")


def test_extract_one_handles_api_failure():
    ext = MethodCardExtractor(api_key="x", base_url="http://x", model="m")
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    ext._client = client  # inject mock; property won't build a real client
    paper = Paper(paper_id="W1", title="T", abstract="A")
    assert ext.extract_one(paper) is None


def test_extract_one_retries_then_parses():
    ext = MethodCardExtractor(api_key="x", base_url="http://x", model="m")

    def reply(text):
        msg = MagicMock()
        msg.choices = [MagicMock(message=MagicMock(content=text))]
        return msg

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        reply("not json"),  # first attempt: unparseable -> retry
        reply('{"task": "top-K rec", "backbone": "LightGCN"}'),  # retry succeeds
    ]
    ext._client = client
    card = ext.extract_one(Paper(paper_id="W2", title="T", abstract="A"))
    assert card is not None
    assert card.paper_id == "W2"
    assert card.task == "top-K rec"
    assert client.chat.completions.create.call_count == 2
