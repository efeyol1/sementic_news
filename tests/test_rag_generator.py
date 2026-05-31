"""Unit tests for src.rag.generator — extractive + grounding guard. No network."""

from __future__ import annotations

from src.rag.context import PromptContext, Snippet
from src.rag.generator import AnthropicGenerator, ExtractiveGenerator, get_generator


def _ctx(snips, *, insufficient=False, lang="tr"):
    return PromptContext(
        canonical="Donald Trump", entity_type="PER", wikidata_qid="Q22686",
        country_code="DE", window_days=30, end_date="2026-05-29", lang=lang,
        top_collocates=[
            {"lemma": "krieg", "pos": "NOUN", "pmi": 3.0, "llr": 50.0},
            {"lemma": "wirtschaft", "pos": "NOUN", "pmi": 2.0, "llr": 20.0},
        ],
        frame_intensities={"conflict": 0.7, "economic": 0.3},
        snippets=snips, insufficient_evidence=insufficient,
        structured_block="...", evidence_block="...",
    )


def _snip(sid, matched):
    return Snippet(id=sid, text="x", source_name="S", date="2026-05-29",
                   link=None, article_id=1, matched_collocates=matched)


def test_extractive_deterministic_and_grounded():
    ctx = _ctx([_snip("S1", ["krieg"]), _snip("S2", ["wirtschaft"])])
    a = ExtractiveGenerator().generate(ctx)
    b = ExtractiveGenerator().generate(ctx)
    assert a.to_payload() == b.to_payload()
    assert a.backend == "extractive"
    valid = {s.id for s in ctx.snippets}
    for claim in a.claims:
        assert claim["citations"]                      # every claim cites
        assert all(c in valid for c in claim["citations"])  # only real ids


def test_extractive_insufficient_no_claims():
    ctx = _ctx([_snip("S1", [])], insufficient=True)
    out = ExtractiveGenerator().generate(ctx)
    assert out.insufficient_evidence is True
    assert out.claims == []


def test_anthropic_grounding_guard_drops_fake_ids():
    ctx = _ctx([_snip("S1", ["krieg"])])
    data = {
        "frame_summary": "x",
        "insufficient_evidence": False,
        "claims": [
            {"text": "real", "citations": ["S1"], "metric_refs": []},
            {"text": "hallucinated", "citations": ["S99"], "metric_refs": []},
            {"text": "metric-only", "citations": [], "metric_refs": ["conflict:0.7"]},
        ],
    }
    out = AnthropicGenerator("claude-sonnet-4-6")._to_explanation(ctx, data)
    texts = [c["text"] for c in out.claims]
    assert "real" in texts           # valid citation kept
    assert "metric-only" in texts    # metric ref → grounded, kept
    assert "hallucinated" not in texts  # fake id dropped (no other grounding)


def test_get_generator_extractive_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {"entity_narrative": {"rag": {"enabled": True}}}
    assert isinstance(get_generator(cfg), ExtractiveGenerator)
