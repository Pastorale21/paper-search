"""Related-Work LLM prompt.

# TODO(D): iterate on this prompt. The goal is a paragraph that
# (a) reads coherently in academic English,
# (b) cites every [N] marker by a real retrieved paper title (no hallucinations),
# (c) groups citations by mechanism rather than chronology where natural,
# (d) ends with a one-sentence comparison to the user's draft idea.
#
# Test loop: change SYSTEM_PROMPT, reload the page (or just the Related Work tab),
# click Generate with the same input + retrieved papers, eyeball the diff.

The page (``ui/pages/4_✍_Related_Work.py``) calls ``build_messages`` and passes the
result to ``openai.OpenAI.chat.completions.create``. ``parse_llm_response`` extracts
the paragraph + the per-citation reference list so the UI can render the fact-check
expander.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert academic writer in graph neural network recommendation. "
    "Given (1) a draft idea or abstract from the user, and (2) a curated list of related "
    "papers with brief method cards, write one coherent related-work paragraph in academic "
    "English. Organize the paragraph by mechanism when possible: task setting, graph "
    "backbone, learning objective, and key idea are more important than chronology. Cite "
    "papers with bracket markers like [1], [2], ... using the exact input order. Every "
    "numbered marker MUST correspond to a real paper from the input list. Do NOT invent "
    "paper titles, paper ids, authors, datasets, or results. If the evidence is not present "
    "in the input, write more generally rather than guessing. Use only citations that are "
    "actually relevant to the sentence where they appear. End with one sentence that "
    "explicitly contrasts the user's idea against the cited line of work.\n\n"
    "Output strictly in this JSON shape:\n"
    "{\n"
    '  "paragraph": "<the related-work paragraph with [N] markers>",\n'
    '  "references": [\n'
    '    {"n": 1, "paper_id": "Wxxx", "one_line_reason": "<why this paper is cited here>"},\n'
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "The references array must include every [N] marker used in the paragraph, and no "
    "uncited papers. If a paper in the input does not fit, omit it; do not pad with "
    "off-topic citations."
)


def build_messages(
    user_input: str,
    retrieved_papers: list[dict],
    target_words: int = 250,
) -> list[dict]:
    """Build the OpenAI chat messages. ``retrieved_papers`` items: ``{paper, method_card}``.

    No network call here — the caller decides when (or whether) to actually invoke chat
    completion. This function is fully testable in isolation.
    """
    lines = [
        f"User draft (paste-as-query, target paragraph length ~{target_words} words):",
        user_input.strip() or "(empty)",
        "",
        "Candidate papers to cite (in order). Use the [N] markers to refer to them:",
    ]
    for n, item in enumerate(retrieved_papers, start=1):
        paper = item.get("paper") or {}
        card = item.get("method_card")
        title = paper.get("title") or "(untitled)"
        year = paper.get("year") or "?"
        pid = paper.get("paper_id") or "?"
        lines.append(f"  [{n}] {title} ({year}, {pid})")
        if card is not None:
            if card.task:
                lines.append(f"      task: {card.task}")
            if card.backbone:
                lines.append(f"      backbone: {card.backbone}")
            if card.loss:
                lines.append(f"      loss: {card.loss}")
            if card.key_idea:
                lines.append(f"      key idea: {card.key_idea}")
        else:
            lines.append("      (no method card on disk — use title/year only)")
        lines.append("")
    user_block = "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_llm_response(raw: str) -> dict:
    """Strip optional ```json fences, parse, default-fill missing keys.

    The page renders ``paragraph`` and walks ``references`` for the fact-check expander.
    A LLM that returns malformed JSON falls back to displaying the raw text.
    """
    import json

    text = _JSON_FENCE_RE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"paragraph": raw, "references": [], "_parse_error": True}
    return {
        "paragraph": data.get("paragraph", ""),
        "references": data.get("references", []),
        "_parse_error": False,
    }


def extract_citation_markers(paragraph: str) -> list[int]:
    """Return all [N] markers found in the paragraph, in order, deduped preserving order."""
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+)\]", paragraph):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
