"""Generation backends for the RAG layer.

Two backends behind one ``Generator`` protocol:

  * ``AnthropicGenerator`` — Claude (model from config), prompt-cached system
    block, structured JSON output. Grounding guard drops any citation to a
    snippet id that wasn't supplied, so a hallucinated id can't leak through.
  * ``ExtractiveGenerator`` — deterministic, no LLM. Builds the same
    ``Explanation`` shape straight from the metrics + retrieved snippets;
    grounded by construction. The fail-soft path (no API key / API error)
    AND a free baseline.

``get_generator(country_config)`` picks Anthropic only when the RAG gate is
on AND ``ANTHROPIC_API_KEY`` is set AND the SDK is importable; otherwise the
extractive fallback. The interface stays open for a future local-LLM backend.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from loguru import logger

from src.rag.config import is_enabled, model_for
from src.rag.context import (
    Explanation,
    PromptContext,
    citations_from_snippets,
    dominant_frame,
    frame_label,
)
from src.rag.prompts import SYSTEM_PROMPT, build_user_block

# 700 truncated Turkish structured-JSON mid-string (TR is more verbose /
# tokenizes longer than EN), causing JSONDecodeError → silent extractive
# fallback. 1500 comfortably fits the 6-claim JSON in any of the 6 languages.
MAX_TOKENS = 1500

# JSON-schema for structured output. Kept within the documented subset
# (objects/arrays/strings/bool, additionalProperties:false, no numeric/length
# constraints).
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "metric_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations", "metric_refs"],
                "additionalProperties": False,
            },
        },
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["frame_summary", "claims", "insufficient_evidence"],
    "additionalProperties": False,
}


class Generator(Protocol):
    backend: str

    def generate(self, ctx: PromptContext) -> Explanation:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Extractive (deterministic, no LLM)
# ---------------------------------------------------------------------------


class ExtractiveGenerator:
    """100%-grounded template generator. Cites only retrieved snippet ids and
    quotes only computed metrics, so it cannot fabricate."""

    backend = "extractive"

    def generate(self, ctx: PromptContext) -> Explanation:
        citations = citations_from_snippets(ctx.snippets)
        dom = dominant_frame(ctx.frame_intensities)
        summary = self._summary(ctx, dom)
        claims = self._claims(ctx)
        insufficient = ctx.insufficient_evidence or not claims
        return Explanation(
            frame_summary=summary,
            claims=claims,
            citations=citations,
            insufficient_evidence=insufficient,
            backend=self.backend,
            model=None,
        )

    def _summary(self, ctx: PromptContext, dom: tuple[str, float] | None) -> str:
        if dom is None:
            if ctx.lang == "tr":
                return (
                    f"{ctx.country_code} medyasında {ctx.canonical} için belirgin "
                    f"bir çerçeve sinyali bulunamadı."
                )
            return (
                f"No clear framing signal for {ctx.canonical} in {ctx.country_code} "
                f"media."
            )
        frame, val = dom
        label = frame_label(frame, ctx.lang)
        if ctx.lang == "tr":
            return (
                f"{ctx.country_code} medyasında {ctx.canonical}, en güçlü "
                f'"{label}" çerçevesiyle ilişkilendirildi (yoğunluk {val:.2f}).'
            )
        return (
            f'In {ctx.country_code} media, {ctx.canonical} was most strongly '
            f'associated with the "{label}" frame (intensity {val:.2f}).'
        )

    def _claims(self, ctx: PromptContext, top_n: int = 3) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for col in ctx.top_collocates[:top_n]:
            lemma = str(col.get("lemma", "")).lower().strip()
            if not lemma:
                continue
            cited = [s.id for s in ctx.snippets if lemma in s.matched_collocates]
            if not cited:
                continue  # no evidence snippet → don't assert it
            pmi = col.get("pmi")
            llr = col.get("llr")
            pmi_s = f"{pmi:.2f}" if isinstance(pmi, (int, float)) else "—"
            llr_s = f"{llr:.1f}" if isinstance(llr, (int, float)) else "—"
            if ctx.lang == "tr":
                text = (
                    f'{ctx.canonical}, {ctx.country_code} medyasında sıkça '
                    f'"{col.get("lemma")}" ile birlikte anıldı '
                    f"(PMI={pmi_s}, LLR={llr_s})."
                )
            else:
                text = (
                    f'In {ctx.country_code} media, {ctx.canonical} frequently '
                    f'co-occurred with "{col.get("lemma")}" '
                    f"(PMI={pmi_s}, LLR={llr_s})."
                )
            claims.append(
                {
                    "text": text,
                    "citations": cited,
                    "metric_refs": [f"PMI:{col.get('lemma')}={pmi_s}"],
                }
            )
        return claims


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicGenerator:
    """Claude-backed generator with prompt caching + structured JSON output.

    Any failure (missing SDK, no key, API error, malformed JSON) downgrades
    to the extractive backend — the caller never sees an exception.
    """

    backend = "anthropic"

    def __init__(self, model: str):
        self.model = model

    def generate(self, ctx: PromptContext) -> Explanation:
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={
                    "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}
                },
                messages=[{"role": "user", "content": build_user_block(ctx)}],
            )
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            data = json.loads(text)
            return self._to_explanation(ctx, data)
        except Exception as exc:  # fail-soft → extractive
            logger.warning(f"AnthropicGenerator → extractive fallback: {exc}")
            return ExtractiveGenerator().generate(ctx)

    def _to_explanation(
        self, ctx: PromptContext, data: dict[str, Any]
    ) -> Explanation:
        """Normalize + GROUNDING GUARD: keep only citations to real snippet
        ids; drop any claim with neither a valid citation nor a metric ref."""
        valid_ids = {s.id for s in ctx.snippets}
        claims: list[dict[str, Any]] = []
        for c in data.get("claims") or []:
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            cited = [cid for cid in (c.get("citations") or []) if cid in valid_ids]
            metric_refs = [str(m) for m in (c.get("metric_refs") or [])]
            if not cited and not metric_refs:
                continue  # ungrounded claim — drop it
            claims.append(
                {"text": text, "citations": cited, "metric_refs": metric_refs}
            )
        insufficient = bool(data.get("insufficient_evidence")) or not claims
        return Explanation(
            frame_summary=str(data.get("frame_summary", "")).strip(),
            claims=claims,
            citations=citations_from_snippets(ctx.snippets),
            insufficient_evidence=insufficient,
            backend=self.backend,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_generator(country_config: dict[str, Any]) -> Generator:
    """Anthropic when the gate is on + a key is present + the SDK imports;
    otherwise the extractive fallback."""
    if is_enabled(country_config) and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401

            return AnthropicGenerator(model_for(country_config))
        except ImportError:
            logger.warning(
                "anthropic SDK not installed — using extractive generator"
            )
    return ExtractiveGenerator()
