"""Prompt templates for the Claude-backed generator.

``SYSTEM_PROMPT`` is static → sent as a prompt-cached block (the dominant
fixed cost). The user block is dynamic (entity context + evidence) and not
cached. The grounding/ethics rules here are load-bearing: they are what
keep the generated text faithful to the evidence and free of country-level
claims.
"""

from __future__ import annotations

from src.rag.context import PromptContext

SYSTEM_PROMPT = """\
You explain PRE-COMPUTED media-framing statistics for one entity in one \
country's news media. You are given (a) computed metrics (top collocates \
with PMI/LLR, frame intensities) and (b) numbered real article snippets \
[S1], [S2], … as evidence.

GROUNDING RULES (strict):
- State ONLY claims supported by the provided metrics or snippets.
- Cite every evidence-based claim with snippet ids, e.g. ["S1","S3"].
- Never invent sources, dates, quotes, numbers, or collocates.
- If fewer than two relevant snippets are provided, or the signal is weak, \
set "insufficient_evidence": true and do not speculate.

ETHICAL RULES (strict):
- This describes ONLY how the sampled media in this country framed this \
entity in this time window. It is NOT a claim about the country, its \
people, or real-world truth.
- Never generalize to a nation ("German media is X", "Poland is Y"). Scope \
every statement to "(this entity, this country's sampled coverage)".
- Do not infer intent, motive, or facts beyond the evidence.

OUTPUT: respond with ONLY a JSON object, no prose around it:
{
  "frame_summary": "<1-2 sentence neutral summary of the framing pattern>",
  "claims": [
    {"text": "<claim>", "citations": ["S1"], "metric_refs": ["conflict:0.62"]}
  ],
  "insufficient_evidence": false
}
Write "frame_summary" and "claims[].text" in the requested output language.\
"""


def build_user_block(ctx: PromptContext) -> str:
    """Dynamic per-entity message: metrics + evidence + language + scope."""
    qid = f" (Wikidata {ctx.wikidata_qid})" if ctx.wikidata_qid else ""
    header = (
        f"Entity: {ctx.canonical}{qid} [{ctx.entity_type}]\n"
        f"Country: {ctx.country_code} | window: last {ctx.window_days} days "
        f"(ending {ctx.end_date})\n"
        f"Output language: {ctx.lang}\n"
    )
    if ctx.insufficient_evidence:
        header += (
            "\nNOTE: evidence is thin — if you cannot ground at least two "
            "claims, return insufficient_evidence: true.\n"
        )
    return (
        f"{header}\n"
        f"=== COMPUTED METRICS ===\n{ctx.structured_block}\n\n"
        f"=== EVIDENCE SNIPPETS ===\n{ctx.evidence_block}\n"
    )
