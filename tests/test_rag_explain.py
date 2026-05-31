"""Unit tests for src.rag.explain — cache hit/miss/stale. No DB/network."""

from __future__ import annotations

from src.rag import explain as explain_mod
from src.rag.context import Explanation, profile_hash


def _profile():
    return {
        "country_code": "DE", "canonical": "Donald Trump", "entity_type": "PER",
        "wikidata_qid": "Q22686", "window_days": 30, "end_date": "2026-05-29",
        "total_mentions": 100, "coverage_days": 27, "avg_pmi": 2.1,
        "avg_log_likelihood": 40.0,
        "top_collocates": [{"lemma": "krieg", "pos": "NOUN", "pmi": 3.0, "llr": 50.0}],
        "frame_intensities": {"conflict": 1.0},
    }


class _FakeGen:
    backend = "extractive"

    def __init__(self):
        self.calls = 0

    def generate(self, ctx):
        self.calls += 1
        return Explanation(
            frame_summary="x", claims=[], citations=[],
            insufficient_evidence=True, backend="extractive", model=None,
        )


def _wire(monkeypatch, gen, cache_row):
    monkeypatch.setattr(explain_mod, "retrieve_snippets", lambda *a, **k: [])
    monkeypatch.setattr(explain_mod, "get_generator", lambda cfg: gen)
    monkeypatch.setattr(explain_mod, "fetch_explanation_cache", lambda *a, **k: cache_row)
    upserts = []
    monkeypatch.setattr(explain_mod, "upsert_explanation_cache", lambda row: upserts.append(row))
    return upserts


def test_cache_hit_skips_generation(monkeypatch):
    p = _profile()
    row = {"profile_hash": profile_hash(p), "explanation": {"frame_summary": "cached"}}
    gen = _FakeGen()
    _wire(monkeypatch, gen, row)
    payload, cached = explain_mod.explain_entity(p, {}, "tr")
    assert cached is True
    assert payload == {"frame_summary": "cached"}
    assert gen.calls == 0


def test_cache_miss_generates_and_upserts(monkeypatch):
    p = _profile()
    gen = _FakeGen()
    upserts = _wire(monkeypatch, gen, None)
    payload, cached = explain_mod.explain_entity(p, {}, "tr")
    assert cached is False
    assert gen.calls == 1
    assert len(upserts) == 1
    assert upserts[0]["profile_hash"] == profile_hash(p)


def test_stale_hash_regenerates(monkeypatch):
    p = _profile()
    stale = {"profile_hash": "deadbeef", "explanation": {"frame_summary": "old"}}
    gen = _FakeGen()
    _wire(monkeypatch, gen, stale)
    payload, cached = explain_mod.explain_entity(p, {}, "tr")
    assert cached is False
    assert gen.calls == 1
