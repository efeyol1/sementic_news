"""Unit tests for src.analysis.text_processing.

These tests cover everything that does NOT require a spaCy model to be
installed: the ``TurkishSurfaceLemmatizer`` (pure Python), the stopword
loader, the ``filter_tokens`` helper, and the ``get_lemmatizer``
factory's cache + dispatch. The spaCy backend's load path is exercised
indirectly by the factory's ``ValueError`` for unsupported languages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import text_processing as tp

# ---------------------------------------------------------------------------
# Stopword loader
# ---------------------------------------------------------------------------


def test_load_stopwords_accepts_bare_list(tmp_path: Path):
    p = tmp_path / "sw.yaml"
    p.write_text("- foo\n- BAR\n- baz\n", encoding="utf-8")
    assert tp.load_stopwords(p) == {"foo", "bar", "baz"}


def test_load_stopwords_accepts_dict_form(tmp_path: Path):
    p = tmp_path / "sw.yaml"
    p.write_text("stopwords:\n  - x\n  - 'y'\n", encoding="utf-8")
    assert tp.load_stopwords(p) == {"x", "y"}


def test_load_stopwords_missing_file_returns_empty(tmp_path: Path):
    assert tp.load_stopwords(tmp_path / "does-not-exist.yaml") == set()


def test_load_stopwords_handles_quoted_on(tmp_path: Path):
    # Regression: bare ``on`` parses as YAML boolean True; quoting fixes it.
    p = tmp_path / "sw.yaml"
    p.write_text("- the\n- 'on'\n- in\n", encoding="utf-8")
    assert tp.load_stopwords(p) == {"the", "on", "in"}


# ---------------------------------------------------------------------------
# Turkish surface-form lemmatizer
# ---------------------------------------------------------------------------


def test_turkish_lemmatizer_strips_common_suffixes():
    lem = tp.TurkishSurfaceLemmatizer()
    toks = lem("Vergileri vergiden vergi.")
    lemmas = [t.lemma for t in toks]
    # Plural+accusative "vergileri" -> "vergi" via -leri strip.
    # Ablative "vergiden" -> "vergi" via -den strip.
    # Bare "vergi" stays "vergi".
    assert "vergi" in lemmas
    assert "vergileri" not in lemmas
    assert "vergiden" not in lemmas
    # All three should collapse to the same stem so they aggregate.
    assert lemmas.count("vergi") == 3


def test_turkish_lemmatizer_respects_min_stem_length():
    lem = tp.TurkishSurfaceLemmatizer()
    # ``ya`` is only 2 chars — must stay as-is rather than disappearing.
    toks = lem("Ya da bu sen")
    assert all(len(t.lemma) >= 1 for t in toks)


def test_turkish_lemmatizer_emits_sentence_ids():
    lem = tp.TurkishSurfaceLemmatizer()
    toks = lem("Birinci cümle. İkinci cümle! Üçüncü cümle?")
    # Three sentences; the splitter increments sent_id after each
    # terminator+whitespace pair.
    sent_ids = sorted({t.sent_id for t in toks})
    assert sent_ids == [0, 1, 2]


def test_turkish_lemmatizer_offsets_are_into_original_string():
    lem = tp.TurkishSurfaceLemmatizer()
    text = "Ankara'da büyük toplantı vardı."
    toks = lem(text)
    # Every token's surface form must be exactly text[start:end].
    for t in toks:
        assert text[t.start : t.end] == t.text


def test_turkish_lemmatizer_handles_empty_string():
    assert tp.TurkishSurfaceLemmatizer()("") == []


# ---------------------------------------------------------------------------
# filter_tokens
# ---------------------------------------------------------------------------


def _tok(lemma: str, pos: str = "NOUN") -> tp.Token:
    return tp.Token(text=lemma, lemma=lemma, pos=pos, start=0, end=len(lemma), sent_id=0)


def test_filter_tokens_drops_stopwords_pos_and_short():
    tokens = [_tok("merkel"), _tok("die", pos="DET"), _tok("ai"), _tok("the")]
    kept = tp.filter_tokens(
        tokens,
        stopwords={"the"},
        keep_pos={"NOUN", "PROPN", "VERB", "ADJ"},
        min_lemma_length=3,
    )
    # "die" drops on POS; "ai" drops on min_lemma_length; "the" drops on
    # stopword; "merkel" survives.
    assert [t.lemma for t in kept] == ["merkel"]


def test_filter_tokens_keep_pos_none_disables_pos_filter():
    tokens = [_tok("merkel", pos="X"), _tok("die", pos="DET")]
    kept = tp.filter_tokens(
        tokens, stopwords=set(), keep_pos=None, min_lemma_length=3
    )
    assert {t.lemma for t in kept} == {"merkel", "die"}


# ---------------------------------------------------------------------------
# get_lemmatizer factory
# ---------------------------------------------------------------------------


def test_get_lemmatizer_returns_turkish_for_tr():
    tp.reset_lemmatizer_cache()
    lem = tp.get_lemmatizer("tr")
    assert isinstance(lem, tp.TurkishSurfaceLemmatizer)


def test_get_lemmatizer_is_cached():
    tp.reset_lemmatizer_cache()
    a = tp.get_lemmatizer("tr")
    b = tp.get_lemmatizer("tr")
    assert a is b


def test_get_lemmatizer_unsupported_language_raises():
    tp.reset_lemmatizer_cache()
    with pytest.raises(ValueError):
        tp.get_lemmatizer("klingon")
