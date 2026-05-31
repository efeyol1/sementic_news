"""Unit tests for src.rag.context — pure, no DB."""

from __future__ import annotations

from src.rag.context import (
    MIN_EVIDENCE_SNIPPETS,
    Snippet,
    build_context,
    dominant_frame,
    profile_hash,
)


def _profile(**over):
    p = {
        "country_code": "DE",
        "canonical": "Donald Trump",
        "entity_type": "PER",
        "wikidata_qid": "Q22686",
        "window_days": 30,
        "end_date": "2026-05-29",
        "total_mentions": 100,
        "coverage_days": 27,
        "avg_pmi": 2.1,
        "avg_log_likelihood": 40.0,
        "top_collocates": [
            {"lemma": "krieg", "pos": "NOUN", "pmi": 3.0, "llr": 50.0},
            {"lemma": "wirtschaft", "pos": "NOUN", "pmi": 2.0, "llr": 20.0},
        ],
        "frame_intensities": {"conflict": 0.7, "economic": 0.3},
    }
    p.update(over)
    return p


def _snip(sid, matched):
    return Snippet(
        id=sid, text="…", source_name="Spiegel", date="2026-05-29",
        link=None, article_id=1, matched_collocates=matched,
    )


def test_profile_hash_stable_under_key_reorder():
    a = _profile()
    b = _profile()
    b["frame_intensities"] = {"economic": 0.3, "conflict": 0.7}  # reordered
    assert profile_hash(a) == profile_hash(b)


def test_profile_hash_changes_on_signal_change():
    a = _profile()
    b = _profile(total_mentions=101)
    assert profile_hash(a) != profile_hash(b)


def test_build_context_blocks_and_ids():
    snips = [_snip("S1", ["krieg"]), _snip("S2", ["wirtschaft"])]
    ctx = build_context(_profile(), snips, "tr")
    assert "krieg" in ctx.structured_block
    assert "conflict" in ctx.structured_block
    assert "[S1]" in ctx.evidence_block
    assert ctx.insufficient_evidence is False


def test_insufficient_when_few_collocate_bearing_snippets():
    snips = [_snip("S1", ["krieg"]), _snip("S2", [])]  # only 1 bearing
    ctx = build_context(_profile(), snips, "en")
    assert MIN_EVIDENCE_SNIPPETS == 2
    assert ctx.insufficient_evidence is True


def test_dominant_frame():
    assert dominant_frame({"conflict": 0.7, "economic": 0.3}) == ("conflict", 0.7)
    assert dominant_frame(None) is None
    assert dominant_frame({"conflict": 0.0}) is None
