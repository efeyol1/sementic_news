"""Unit tests for src.analysis.frame_bridge (Sprint 10).

No DB, no spaCy — the bridge is pure Python over the real frame lexicons
in ``configs/frames/*.yaml`` plus hand-built collocate dicts.

Coverage:
  * lexicon loading (sets, lowercased, cached, missing-file)
  * intensity: single/multi match, L1 normalization, LLR weighting
  * honest empties: no match → None, empty lexicon → None, zero weight
  * case-insensitive matching
  * config gate (is_enabled)
"""

from __future__ import annotations

import pytest

from src.analysis import frame_bridge as fb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collocate(lemma: str, llr: float, *, pmi: float = 1.0, pos: str = "NOUN") -> dict:
    return {"lemma": lemma, "pos": pos, "c11_window": 5, "pmi": pmi, "llr": llr}


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------


def test_load_frame_lexicon_returns_six_frames_as_sets():
    lex = fb.load_frame_lexicon("en")
    assert set(lex.keys()) == set(fb.FRAME_NAMES)
    for frame, seeds in lex.items():
        assert isinstance(seeds, set)
        assert seeds  # non-empty


def test_load_frame_lexicon_lowercases_seeds():
    lex = fb.load_frame_lexicon("de")
    # German lexicon entries are lowercased at load so they match the
    # lowercased lemmas spaCy emits.
    for seeds in lex.values():
        assert all(s == s.lower() for s in seeds)


def test_load_frame_lexicon_missing_language_returns_empty():
    # Unknown language → empty dict, no raise.
    fb._LEXICON_CACHE.pop("zz", None)
    assert fb.load_frame_lexicon("zz") == {}


def test_load_frame_lexicon_is_cached():
    a = fb.load_frame_lexicon("fr")
    b = fb.load_frame_lexicon("fr")
    assert a is b  # same object returned from cache


# ---------------------------------------------------------------------------
# Intensity computation
# ---------------------------------------------------------------------------


def test_single_match_yields_normalized_distribution():
    # "war" is a conflict seed (en). Single match → conflict == 1.0,
    # every other frame 0.0, and the dict sums to 1.0.
    out = fb.compute_frame_intensities([_collocate("war", 40.0)], "en")
    assert out is not None
    assert out["conflict"] == pytest.approx(1.0)
    assert sum(out.values()) == pytest.approx(1.0)
    assert set(out.keys()) == set(fb.FRAME_NAMES)


def test_llr_weighting_splits_intensity_proportionally():
    # economy (economic, llr=30) vs war (conflict, llr=10) → 0.75 / 0.25.
    out = fb.compute_frame_intensities(
        [_collocate("economy", 30.0), _collocate("war", 10.0)], "en"
    )
    assert out is not None
    assert out["economic"] == pytest.approx(0.75)
    assert out["conflict"] == pytest.approx(0.25)
    assert sum(out.values()) == pytest.approx(1.0)


def test_case_insensitive_match():
    out = fb.compute_frame_intensities([_collocate("WAR", 12.0)], "en")
    assert out is not None
    assert out["conflict"] == pytest.approx(1.0)


def test_no_match_returns_none():
    out = fb.compute_frame_intensities(
        [_collocate("zzzznonsense", 50.0)], "en"
    )
    assert out is None


def test_empty_lexicon_returns_none():
    out = fb.compute_frame_intensities([_collocate("war", 40.0)], "en", lexicon={})
    assert out is None


def test_zero_weight_match_returns_none():
    # A matched collocate whose LLR is 0 carries no signal → no fabricated
    # distribution.
    out = fb.compute_frame_intensities([_collocate("war", 0.0)], "en")
    assert out is None


def test_negative_llr_clamped_to_zero():
    out = fb.compute_frame_intensities(
        [_collocate("war", -5.0), _collocate("economy", 10.0)], "en"
    )
    assert out is not None
    # war clamped to 0 → all weight on economic.
    assert out["economic"] == pytest.approx(1.0)
    assert out["conflict"] == pytest.approx(0.0)


def test_explicit_lexicon_overrides_language_file():
    lex = {"economic": {"foo"}, "security": {"bar"}}
    out = fb.compute_frame_intensities(
        [_collocate("foo", 10.0), _collocate("bar", 10.0)], "en", lexicon=lex
    )
    assert out is not None
    assert out == {"economic": pytest.approx(0.5), "security": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def test_is_enabled_true_when_flag_set():
    cfg = {"entity_narrative": {"frame_bridge": {"enabled": True}}}
    assert fb.is_enabled(cfg) is True


def test_is_enabled_defaults_false():
    assert fb.is_enabled({"entity_narrative": {}}) is False
    assert fb.is_enabled({}) is False
