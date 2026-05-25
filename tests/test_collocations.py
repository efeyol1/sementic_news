"""Unit tests for src.analysis.collocations.

Strategy: pre-built ``Token`` streams + a stub lemmatizer + monkeypatched
DB fetch/upsert. No spaCy required.

The most important contract pinned here is the *position offset*:
``entity_mentions.position_in_article`` is the HuggingFace NER ``start``
offset into :func:`src.analysis.text_inputs.build_ner_text`'s output —
NOT into ``cleaned_article_text`` alone. The collocations module reuses
``build_ner_text`` so windows line up; if anyone breaks that contract,
``test_position_matches_build_ner_text`` will fail.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.analysis import collocations as col
from src.analysis import text_processing as tp
from src.analysis.text_inputs import build_ner_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeLemmatizer:
    """Deterministic whitespace lemmatizer with explicit per-token POS."""

    def __init__(self, mapping: dict[str, list[tp.Token]]):
        self._mapping = mapping
        self.calls: list[str] = []
        self.language = "test"

    def __call__(self, text: str) -> list[tp.Token]:
        self.calls.append(text)
        return list(self._mapping.get(text, []))


def _toks(*specs: tuple[str, str, str, int, int, int]) -> list[tp.Token]:
    """Build a token list. Spec = (text, lemma, pos, start, end, sent_id)."""
    return [tp.Token(*spec) for spec in specs]


def _row(**overrides: Any) -> dict[str, Any]:
    """Default mention row joined to its article."""
    base = {
        "mention_id": 1,
        "article_id": 1,
        "entity_text": "Trump",
        "entity_type": "PER",
        "wikidata_qid": "Q22686",
        "canonical": "Donald Trump",
        "position_in_article": 0,
        "country_code": "DE",
        "collected_date": "2026-05-25",
        "title": "",
        "summary": "",
        "article_text": "",
        "cleaned_title": "",
        "cleaned_summary": "",
        "cleaned_article_text": "",
    }
    base.update(overrides)
    return base


def _country_config(
    *,
    enabled: bool = True,
    window: int = 5,
    respect_sentence_boundary: bool = True,
    keep_pos: list[str] | None = None,
    min_lemma_length: int = 3,
    min_cooccurrence_count: int = 1,
    language: str = "en",
) -> dict[str, Any]:
    return {
        "country_code": "DE",
        "country_slug": "germany",
        "language": language,
        "entity_narrative": {
            "enabled": True,
            "collocations": {
                "enabled": enabled,
                "window": window,
                "respect_sentence_boundary": respect_sentence_boundary,
                "keep_pos": (
                    keep_pos if keep_pos is not None
                    else ["NOUN", "PROPN", "VERB", "ADJ"]
                ),
                "min_lemma_length": min_lemma_length,
                "min_cooccurrence_count": min_cooccurrence_count,
                "lemmatizer": "spacy",
            },
        },
    }


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def test_is_enabled_defaults_to_false():
    assert col.is_enabled({}) is False
    assert col.is_enabled({"entity_narrative": {}}) is False
    cfg = {"entity_narrative": {"collocations": {"enabled": True}}}
    assert col.is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# _find_mention_span — character offset → token span
# ---------------------------------------------------------------------------


def test_find_mention_span_single_token():
    # Tokens: "Trump"(0..5), "won"(6..9), "again"(10..15)
    tokens = _toks(
        ("Trump", "trump", "PROPN", 0, 5, 0),
        ("won", "win", "VERB", 6, 9, 0),
        ("again", "again", "ADV", 10, 15, 0),
    )
    first, last = col._find_mention_span(tokens, position=0, entity_text="Trump")
    assert (first, last) == (0, 0)


def test_find_mention_span_multi_token():
    # "Donald Trump" spans two tokens.
    tokens = _toks(
        ("Donald", "donald", "PROPN", 0, 6, 0),
        ("Trump", "trump", "PROPN", 7, 12, 0),
        ("won", "win", "VERB", 13, 16, 0),
    )
    first, last = col._find_mention_span(tokens, position=0, entity_text="Donald Trump")
    assert (first, last) == (0, 1)


def test_find_mention_span_no_match_returns_minus_one():
    tokens = _toks(("X", "x", "X", 0, 1, 0))
    first, last = col._find_mention_span(tokens, position=100, entity_text="Trump")
    assert (first, last) == (-1, -1)


# ---------------------------------------------------------------------------
# _window_tokens
# ---------------------------------------------------------------------------


def test_window_tokens_excludes_mention_itself():
    tokens = _toks(
        ("a", "a", "X", 0, 1, 0),
        ("b", "b", "X", 2, 3, 0),
        ("Trump", "trump", "PROPN", 4, 9, 0),
        ("c", "c", "X", 10, 11, 0),
        ("d", "d", "X", 12, 13, 0),
    )
    out = col._window_tokens(tokens, first_idx=2, last_idx=2, window=10, respect_sentence=False)
    assert [t.lemma for t in out] == ["a", "b", "c", "d"]


def test_window_tokens_respects_sentence_boundary():
    # Anchor in sentence 0; tokens in sentence 1 are excluded.
    tokens = _toks(
        ("a", "a", "X", 0, 1, 0),
        ("Trump", "trump", "PROPN", 2, 7, 0),
        ("b", "b", "X", 8, 9, 0),    # same sentence
        ("c", "c", "X", 10, 11, 1),  # next sentence — excluded
        ("d", "d", "X", 12, 13, 1),  # next sentence — excluded
    )
    out = col._window_tokens(tokens, 1, 1, window=10, respect_sentence=True)
    assert [t.lemma for t in out] == ["a", "b"]


def test_window_tokens_n_limits_count_each_side():
    tokens = _toks(
        *[(str(i), str(i), "X", i, i + 1, 0) for i in range(10)],
    )
    # Anchor at index 5; window=2 → left = [3,4], right = [6,7].
    out = col._window_tokens(tokens, 5, 5, window=2, respect_sentence=False)
    assert [t.lemma for t in out] == ["3", "4", "6", "7"]


def test_window_tokens_no_anchor_returns_empty():
    tokens = _toks(("x", "x", "X", 0, 1, 0))
    assert col._window_tokens(tokens, -1, -1, 5, True) == []


# ---------------------------------------------------------------------------
# build_ner_text offset contract — the critical wrinkle
# ---------------------------------------------------------------------------


def test_position_matches_build_ner_text():
    """``entity_mentions.position_in_article`` is the HF NER ``start``
    offset into ``build_ner_text(item)``, not into ``cleaned_article_text``
    alone. If anyone changes ``build_ner_text``'s concatenation rule
    without updating this module, the offsets diverge silently."""
    item = {
        "cleaned_title": "Merkel meets Trump",
        "cleaned_summary": "Bilateral talks held",
        "cleaned_article_text": "Trump arrived in Berlin today.",
    }
    text = build_ner_text(item)
    # Concatenation rule: parts joined by ". " — title + ". " + summary + ". " + body.
    assert text == (
        "Merkel meets Trump. Bilateral talks held. "
        "Trump arrived in Berlin today."
    )
    # The mention "Trump" appears at offset text.find("Trump") — that's
    # exactly the value the NER pipeline would write to position_in_article.
    pos = text.find("Trump")
    assert pos == 13
    assert text[pos : pos + 5] == "Trump"


# ---------------------------------------------------------------------------
# End-to-end: extract_collocations_batch with monkeypatched DB
# ---------------------------------------------------------------------------


def _stub_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fetch_rows: list[dict[str, Any]],
    fake_lemmatizer: _FakeLemmatizer,
) -> dict[str, Any]:
    captured: dict[str, Any] = {"upserted": None}

    monkeypatch.setattr(
        col, "fetch_for_collocation_extraction",
        lambda date_str, country_code: list(fetch_rows),
    )

    def fake_upsert(rows, country_code, date_str):
        captured["upserted"] = list(rows)
        captured["country_code"] = country_code
        captured["date_str"] = date_str
        return len(rows)

    monkeypatch.setattr(col, "bulk_upsert_entity_collocations", fake_upsert)
    monkeypatch.setattr(col, "get_lemmatizer", lambda *a, **k: fake_lemmatizer)
    return captured


def test_extract_collocations_batch_skips_when_disabled(monkeypatch):
    cfg = _country_config(enabled=False)
    captured: dict[str, Any] = {"upserted": "NEVER"}
    monkeypatch.setattr(
        col, "fetch_for_collocation_extraction",
        lambda *a, **k: pytest.fail("DB fetch must not run when disabled"),
    )
    monkeypatch.setattr(
        col, "bulk_upsert_entity_collocations",
        lambda *a, **k: pytest.fail("DB write must not run when disabled"),
    )
    assert col.extract_collocations_batch(
        date_str="2026-05-25", country_config=cfg
    ) == 0
    assert captured["upserted"] == "NEVER"


def test_extract_collocations_batch_aggregates_and_writes(monkeypatch, tmp_path):
    # Build the same string the lemmatizer will receive.
    item_text_part = "Trump talked tariffs today"
    item = {
        "cleaned_title": item_text_part,
        "cleaned_summary": "",
        "cleaned_article_text": "",
    }
    text = build_ner_text(item)
    assert text == item_text_part  # body+summary empty, no leading ". "

    tokens = _toks(
        ("Trump",   "trump",  "PROPN", 0, 5,   0),
        ("talked",  "talk",   "VERB",  6, 12,  0),
        ("tariffs", "tariff", "NOUN", 13, 20,  0),
        ("today",   "today",  "NOUN", 21, 26,  0),
    )
    fake_lem = _FakeLemmatizer({text: tokens})

    row = _row(
        cleaned_title=item_text_part,
        cleaned_summary="",
        cleaned_article_text="",
        position_in_article=0,
        entity_text="Trump",
        canonical="Donald Trump",
        wikidata_qid="Q22686",
    )
    captured = _stub_pipeline(monkeypatch, fetch_rows=[row], fake_lemmatizer=fake_lem)

    cfg = _country_config(window=10, min_cooccurrence_count=1)
    n = col.extract_collocations_batch(
        date_str="2026-05-25", country_config=cfg
    )
    assert n == 3
    upserted = {r["lemma"]: r for r in captured["upserted"]}
    # "trump" excluded (mention itself); "today" included (NOUN); "talk"
    # included (VERB); "tariff" included (NOUN).
    assert set(upserted) == {"talk", "tariff", "today"}
    for lemma, expected_pos in [
        ("talk", "VERB"), ("tariff", "NOUN"), ("today", "NOUN"),
    ]:
        assert upserted[lemma]["cooccurrence_count"] == 1
        assert upserted[lemma]["pos"] == expected_pos
        assert upserted[lemma]["canonical"] == "Donald Trump"
        assert upserted[lemma]["wikidata_qid"] == "Q22686"
        assert upserted[lemma]["entity_type"] == "PER"


def test_extract_collocations_batch_applies_min_cooccurrence(monkeypatch):
    item_text = "Trump tariff. Trump tariff. Trump policy."
    item = {"cleaned_title": item_text, "cleaned_summary": "", "cleaned_article_text": ""}
    text = build_ner_text(item)
    assert text == item_text

    # Hand-craft tokens with stable offsets matching the string positions.
    # Sentences split on '. ' so sent_id increments accordingly.
    tokens = [
        tp.Token("Trump",   "trump",   "PROPN", 0,  5,  0),
        tp.Token("tariff",  "tariff",  "NOUN",  6,  12, 0),
        tp.Token("Trump",   "trump",   "PROPN", 14, 19, 1),
        tp.Token("tariff",  "tariff",  "NOUN",  20, 26, 1),
        tp.Token("Trump",   "trump",   "PROPN", 28, 33, 2),
        tp.Token("policy",  "policy",  "NOUN",  34, 40, 2),
    ]
    fake_lem = _FakeLemmatizer({text: tokens})

    # Three mentions, one per sentence, anchored at the corresponding offset.
    rows = [
        _row(
            mention_id=i + 1,
            position_in_article=pos,
            cleaned_title=item_text,
            cleaned_summary="",
            cleaned_article_text="",
        )
        for i, pos in enumerate([0, 14, 28])
    ]
    captured = _stub_pipeline(monkeypatch, fetch_rows=rows, fake_lemmatizer=fake_lem)

    cfg = _country_config(window=10, min_cooccurrence_count=2,
                          respect_sentence_boundary=True)
    col.extract_collocations_batch(date_str="2026-05-25", country_config=cfg)
    upserted = {r["lemma"]: r["cooccurrence_count"] for r in captured["upserted"]}
    # tariff appears twice → kept; policy appears once → dropped.
    assert upserted == {"tariff": 2}


def test_extract_collocations_batch_empty_rows_still_clears(monkeypatch):
    fake_lem = _FakeLemmatizer({})
    captured = _stub_pipeline(monkeypatch, fetch_rows=[], fake_lemmatizer=fake_lem)
    cfg = _country_config()
    n = col.extract_collocations_batch(
        date_str="2026-05-25", country_config=cfg
    )
    assert n == 0
    # ``bulk_upsert_entity_collocations`` was called with an empty list
    # so the DELETE-side of the idempotency contract still fired.
    assert captured["upserted"] == []
    assert captured["country_code"] == "DE"
    assert captured["date_str"] == "2026-05-25"


def test_extract_collocations_batch_skips_invalid_position(monkeypatch):
    item_text = "Hello world."
    item = {"cleaned_title": item_text, "cleaned_summary": "", "cleaned_article_text": ""}
    text = build_ner_text(item)
    tokens = _toks(("Hello", "hello", "PROPN", 0, 5, 0),
                   ("world", "world", "NOUN", 6, 11, 0))
    fake_lem = _FakeLemmatizer({text: tokens})

    # position beyond the text length → mention is skipped, no rows.
    row = _row(position_in_article=999, cleaned_title=item_text,
               cleaned_summary="", cleaned_article_text="")
    captured = _stub_pipeline(monkeypatch, fetch_rows=[row], fake_lemmatizer=fake_lem)
    cfg = _country_config()
    col.extract_collocations_batch(date_str="2026-05-25", country_config=cfg)
    assert captured["upserted"] == []
