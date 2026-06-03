"""Citation intent classification into {background, method, comparison}.

Dispatch order (see nlp/HANDOFF.md):
  1. USE_SCICITE_MODEL flag -> B's SciBERT/SciCite local model.
  2. Semantic Scholar pre-classified `intents` on the context dict -> label mapping.
  3. LLM zero-shot fallback (works today; the live path while data source is OpenAlex).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from .. import config

Intent = Literal["background", "method", "comparison"]

# Semantic Scholar's citation-intent labels -> our 3-class space.
_S2_LABEL_MAP: dict[str, Intent] = {
    "background": "background",
    "methodology": "method",
    "method": "method",
    "result": "comparison",
}

_SCICITE_LABEL_MAP: dict[str, Intent] = {
    "background": "background",
    "method": "method",
    "result": "comparison",
    "comparison": "comparison",
    "label_0": "background",
    "label_1": "method",
    "label_2": "comparison",
}

_LLM_SYSTEM = (
    "You classify a citation context (the sentence where one paper cites another) into "
    "exactly one of three intents:\n"
    "- background: cited for general context, motivation, or prior knowledge.\n"
    "- method: the cited work's method/technique/dataset is used or built upon.\n"
    "- comparison: the cited work is compared against (a baseline or result contrast).\n"
    "Reply with ONLY the single lowercase label word and nothing else."
)
_LLM_ONE_SHOT_USER = "Citation context: We adopt the BPR loss of [CITATION] to train our model."
_LLM_ONE_SHOT_ASSISTANT = "method"


def map_s2_intent(label: str) -> Intent:
    """Map a Semantic Scholar intent label to our 3-class space (unknown -> background)."""
    return _S2_LABEL_MAP.get((label or "").strip().lower(), "background")


def map_scicite_label(label: str) -> Intent:
    """Map a SciCite/model label to our 3-class space (unknown -> background)."""
    normalized = (label or "").strip().lower().replace("-", "_")
    return _SCICITE_LABEL_MAP.get(normalized, "background")


def _extract_text(context) -> str:
    """Pull the citation-context sentence out of a str or an S2-style context dict."""
    if isinstance(context, dict):
        return str(context.get("context") or context.get("text") or "")
    return str(context or "")


@lru_cache(maxsize=1)
def _get_scicite_pipeline():
    """Load the local SciCite classifier pipeline, failing clearly if it is not trained."""
    model_dir = config.SCICITE_MODEL_DIR
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"SciCite model not found at {model_dir}. "
            "Train it with: uv run python -m nlp.citation_intent.train_scicite --epochs 3"
        )
    from transformers import pipeline

    return pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir))


def classify_with_scicite(context) -> Intent:
    """Classify via a local SciBERT model fine-tuned on SciCite."""
    text = _extract_text(context)
    if not text.strip():
        return "background"
    result = _get_scicite_pipeline()(text, truncation=True)
    prediction = result[0] if isinstance(result, list) else result
    return map_scicite_label(str(prediction.get("label", "")))


class CitationIntentClassifier:
    """Classify citation contexts into {background, method, comparison}."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        use_scicite: bool | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = base_url if base_url is not None else config.LLM_BASE_URL
        self.model = model if model is not None else config.LLM_MODEL
        self.use_scicite = config.USE_SCICITE_MODEL if use_scicite is None else use_scicite
        self._client = None  # lazily constructed so tests need no real key

    @property
    def client(self):
        """Lazily build the OpenAI-compatible client (deferred for the LLM fallback)."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def classify(self, context) -> Intent:
        """Classify a citation context (str, or S2-style dict with an `intents` field)."""
        if self.use_scicite:
            return classify_with_scicite(context)  # type: ignore[return-value]
        if isinstance(context, dict) and context.get("intents"):
            return map_s2_intent(context["intents"][0])
        return self._classify_with_llm(_extract_text(context))

    def _classify_with_llm(self, text: str) -> Intent:
        """Zero-/one-shot LLM classification; defaults to 'background' on any ambiguity."""
        if not text.strip():
            return "background"
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": _LLM_ONE_SHOT_USER},
                {"role": "assistant", "content": _LLM_ONE_SHOT_ASSISTANT},
                {"role": "user", "content": f"Citation context: {text}"},
            ],
        )
        reply = (resp.choices[0].message.content or "").strip().lower()
        for label in ("comparison", "method", "background"):
            if label in reply:
                return label  # type: ignore[return-value]
        return "background"
