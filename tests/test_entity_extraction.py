from __future__ import annotations

import pytest


class _FakePipe:
    """Mimics a HuggingFace ``aggregation_strategy='simple'`` NER pipeline."""

    def __init__(self, by_text: dict[str, list[dict]]):
        self._by_text = by_text
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[dict]:
        self.calls.append(text)
        return list(self._by_text.get(text, []))


def test_is_enabled_defaults_to_false():
    from src.analysis import entity_extraction

    assert entity_extraction.is_enabled({}) is False
    assert entity_extraction.is_enabled({"entity_narrative": {}}) is False
    assert entity_extraction.is_enabled(
        {"entity_narrative": {"enabled": False}}
    ) is False
    assert entity_extraction.is_enabled(
        {"entity_narrative": {"enabled": True}}
    ) is True


def test_normalize_label_strips_bio_prefix():
    from src.analysis.entity_extraction import _normalize_label

    assert _normalize_label("PER") == "PER"
    assert _normalize_label("B-PER") == "PER"
    assert _normalize_label("I-ORG") == "ORG"
    assert _normalize_label("b-loc") == "LOC"


def test_extract_mentions_filters_by_keep_labels():
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe(
        {
            "Merkel besuchte Berlin.": [
                {"entity_group": "PER", "word": "Merkel", "start": 0, "end": 6},
                {"entity_group": "LOC", "word": "Berlin", "start": 17, "end": 23},
                {"entity_group": "MISC", "word": "CDU", "start": 30, "end": 33},
            ],
        }
    )
    mentions = _extract_mentions(
        "Merkel besuchte Berlin.", pipe, keep_labels={"PER", "LOC"}
    )

    assert mentions == [
        {"entity_text": "Merkel", "entity_type": "PER", "position_in_article": 0},
        {"entity_text": "Berlin", "entity_type": "LOC", "position_in_article": 17},
    ]


def test_extract_mentions_drops_blank_words_and_unknown_labels():
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe(
        {
            "x": [
                {"entity_group": "PER", "word": "   ", "start": 0},
                {"entity_group": "DATE", "word": "Mittwoch", "start": 5},
                {"entity_group": "B-ORG", "word": "Bundestag", "start": 10},
            ],
        }
    )
    mentions = _extract_mentions("x", pipe, keep_labels={"PER", "ORG", "LOC", "MISC"})

    assert mentions == [
        {"entity_text": "Bundestag", "entity_type": "ORG", "position_in_article": 10},
    ]


def test_extract_mentions_returns_empty_for_blank_text():
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe({})
    assert _extract_mentions("", pipe, keep_labels={"PER"}) == []


def test_extract_mentions_filters_subword_artifacts():
    """Regression: 2026-05-16 DE smoke run leaked ``##i jinping`` and ``mer``
    (Friedrich Merz'in WordPiece parçası). ``aggregation_strategy="average"``
    çoğu artifact'i birleştiriyor ama yine de ``##`` ile başlayan ya da
    bileşeni ``##`` olan tokenlar sızabiliyor — bunlar reddedilmeli."""
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe(
        {
            "x": [
                {"entity_group": "PER", "word": "##i jinping", "start": 0, "score": 0.99},
                {"entity_group": "PER", "word": "##wort", "start": 10, "score": 0.95},
                {"entity_group": "PER", "word": "Friedrich Merz", "start": 20, "score": 0.97},
            ],
        }
    )
    mentions = _extract_mentions("x", pipe, keep_labels={"PER"})
    assert [m["entity_text"] for m in mentions] == ["Friedrich Merz"]


def test_extract_mentions_filters_short_fragments():
    """``"j"``, ``"al"``, ``"x"`` gibi tek-iki harfli fragmentler tokenizer
    gürültüsü; canonical kişi/yer ismi olarak kabul edilemez. 3-char gate
    Arapça ``al-`` artikellerinin standalone yakalandığı DE smoke run
    sorununu çözer."""
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe(
        {
            "y": [
                {"entity_group": "PER", "word": "j", "start": 0, "score": 0.99},
                {"entity_group": "PER", "word": "AL", "start": 5, "score": 0.99},
                {"entity_group": "PER", "word": "Xi", "start": 10, "score": 0.99},
                {"entity_group": "LOC", "word": "Berlin", "start": 15, "score": 0.99},
            ],
        }
    )
    mentions = _extract_mentions("y", pipe, keep_labels={"PER", "LOC"})
    assert [m["entity_text"] for m in mentions] == ["Berlin"]


def test_extract_mentions_filters_low_confidence():
    """``aggregation_strategy="average"`` her span'a aggregated ``score``
    bırakır. Eşik altı (default 0.5) tahminler collocation profiline
    girmemeli — yanlış canonical sızması Sprint 5 PMI'yı bozar. Davlan
    multilingual NER tipik olarak 0.6–0.85 skorluyor; eşik 0.5 sadece
    "model emin değil" vakalarını eler."""
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe(
        {
            "z": [
                {"entity_group": "PER", "word": "Solid Person", "start": 0, "score": 0.95},
                {"entity_group": "PER", "word": "Borderline Person", "start": 13, "score": 0.62},
                {"entity_group": "PER", "word": "Reject Me", "start": 31, "score": 0.42},
                {"entity_group": "PER", "word": "Also Reject", "start": 41, "score": 0.30},
            ],
        }
    )
    mentions = _extract_mentions("z", pipe, keep_labels={"PER"})
    assert [m["entity_text"] for m in mentions] == ["Solid Person", "Borderline Person"]


def test_extract_mentions_keeps_entities_without_score_field():
    """Eski test fixture'ları ve bazı pipeline versiyonları ``score`` field'ı
    döndürmüyor; eşik atlanmalı, yoksa unit testler ve eski cache yolları
    bozulur. Sadece score *verildiğinde* gate uygulanır."""
    from src.analysis.entity_extraction import _extract_mentions

    pipe = _FakePipe(
        {
            "q": [
                {"entity_group": "PER", "word": "Angela Merkel", "start": 0},
            ],
        }
    )
    mentions = _extract_mentions("q", pipe, keep_labels={"PER"})
    assert [m["entity_text"] for m in mentions] == ["Angela Merkel"]


def test_extract_entities_batch_skips_when_disabled(monkeypatch):
    from src.analysis import entity_extraction

    monkeypatch.setattr(
        entity_extraction,
        "fetch_for_entity_extraction",
        lambda *a, **kw: pytest.fail("DB should not be touched when disabled"),
    )
    monkeypatch.setattr(
        entity_extraction,
        "_load_pipeline",
        lambda *a, **kw: pytest.fail("model should not load when disabled"),
    )

    inserted = entity_extraction.extract_entities_batch(
        date_str="2026-05-14",
        country_config={
            "country_code": "TR",
            "country_slug": "turkey",
            "entity_narrative": {"enabled": False},
        },
    )
    assert inserted == 0


def test_extract_entities_batch_skips_when_no_items(monkeypatch):
    from src.analysis import entity_extraction

    monkeypatch.setattr(
        entity_extraction,
        "fetch_for_entity_extraction",
        lambda date_str, country_code: [],
    )
    monkeypatch.setattr(
        entity_extraction,
        "_load_pipeline",
        lambda model_id: pytest.fail("model should not load with no items"),
    )

    inserted = entity_extraction.extract_entities_batch(
        date_str="2026-05-14",
        country_config={
            "country_code": "DE",
            "country_slug": "germany",
            "entity_narrative": {"enabled": True},
        },
    )
    assert inserted == 0


def test_extract_entities_batch_inserts_mentions_with_provenance(monkeypatch):
    from src.analysis import entity_extraction

    items = [
        {
            "id": 1,
            "title": "Merkel besuchte Berlin.",
            "summary": "",
            "article_text": "",
            "cleaned_article_text": "",
            "cleaned_title": "Merkel besuchte Berlin.",
            "cleaned_summary": "",
        },
        {
            "id": 2,
            "title": "Macron kommt nach Paris.",
            "summary": "",
            "article_text": "",
            "cleaned_article_text": "",
            "cleaned_title": "Macron kommt nach Paris.",
            "cleaned_summary": "",
        },
    ]
    monkeypatch.setattr(
        entity_extraction,
        "fetch_for_entity_extraction",
        lambda date_str, country_code: items,
    )

    pipe = _FakePipe(
        {
            "Merkel besuchte Berlin.": [
                {"entity_group": "PER", "word": "Merkel", "start": 0},
                {"entity_group": "LOC", "word": "Berlin", "start": 17},
            ],
            "Macron kommt nach Paris.": [
                {"entity_group": "PER", "word": "Macron", "start": 0},
                {"entity_group": "LOC", "word": "Paris", "start": 18},
            ],
        }
    )
    monkeypatch.setattr(entity_extraction, "_load_pipeline", lambda model_id: pipe)

    captured: dict[str, object] = {}

    def fake_bulk_insert(mentions, article_ids):
        captured["mentions"] = list(mentions)
        captured["article_ids"] = list(article_ids)
        return len(mentions)

    monkeypatch.setattr(
        entity_extraction, "bulk_insert_entity_mentions", fake_bulk_insert
    )

    inserted = entity_extraction.extract_entities_batch(
        date_str="2026-05-14",
        country_config={
            "country_code": "DE",
            "country_slug": "germany",
            "entity_narrative": {
                "enabled": True,
                "ner_model": "Davlan/bert-base-multilingual-cased-ner-hrl",
            },
        },
    )

    assert inserted == 4
    assert captured["article_ids"] == [1, 2]
    mentions = captured["mentions"]
    assert {m["article_id"] for m in mentions} == {1, 2}
    assert all(m["country_code"] == "DE" for m in mentions)
    assert all(m["collected_date"] == "2026-05-14" for m in mentions)
    types_per_article = {
        m["article_id"]: sorted(x["entity_type"] for x in mentions if x["article_id"] == m["article_id"])
        for m in mentions
    }
    assert types_per_article[1] == ["LOC", "PER"]
    assert types_per_article[2] == ["LOC", "PER"]


def test_extract_entities_batch_passes_keep_labels_override(monkeypatch):
    from src.analysis import entity_extraction

    monkeypatch.setattr(
        entity_extraction,
        "fetch_for_entity_extraction",
        lambda date_str, country_code: [
            {
                "id": 1,
                "title": "Merkel CDU Berlin.",
                "summary": "",
                "article_text": "",
                "cleaned_article_text": "",
                "cleaned_title": "Merkel CDU Berlin.",
                "cleaned_summary": "",
            }
        ],
    )

    pipe = _FakePipe(
        {
            "Merkel CDU Berlin.": [
                {"entity_group": "PER", "word": "Merkel", "start": 0},
                {"entity_group": "MISC", "word": "CDU", "start": 7},
                {"entity_group": "LOC", "word": "Berlin", "start": 11},
            ],
        }
    )
    monkeypatch.setattr(entity_extraction, "_load_pipeline", lambda model_id: pipe)

    captured: dict[str, object] = {}

    def fake_bulk_insert(mentions, article_ids):
        captured["mentions"] = list(mentions)
        return len(mentions)

    monkeypatch.setattr(
        entity_extraction, "bulk_insert_entity_mentions", fake_bulk_insert
    )

    entity_extraction.extract_entities_batch(
        date_str="2026-05-14",
        country_config={
            "country_code": "DE",
            "country_slug": "germany",
            "entity_narrative": {
                "enabled": True,
                # Restrict to PER + LOC; MISC ("CDU") must be filtered out.
                "keep_labels": ["PER", "LOC"],
            },
        },
    )

    types = sorted(m["entity_type"] for m in captured["mentions"])
    assert types == ["LOC", "PER"]


def test_extract_entities_batch_applies_local_aliases(monkeypatch):
    from src.analysis import entity_extraction

    monkeypatch.setattr(
        entity_extraction,
        "fetch_for_entity_extraction",
        lambda date_str, country_code: [
            {
                "id": 1,
                "title": "Cumhurbaşkanı Erdoğan açıklama yaptı.",
                "summary": "",
                "article_text": "",
                "cleaned_article_text": "",
                "cleaned_title": "Cumhurbaşkanı Erdoğan açıklama yaptı.",
                "cleaned_summary": "",
            }
        ],
    )
    pipe = _FakePipe(
        {
            "Cumhurbaşkanı Erdoğan açıklama yaptı.": [
                {"entity_group": "PER", "word": "Cumhurbaşkanı Erdoğan", "start": 0},
            ],
        }
    )
    monkeypatch.setattr(entity_extraction, "_load_pipeline", lambda model_id: pipe)

    captured: dict[str, object] = {}

    def fake_bulk_insert(mentions, article_ids):
        captured["mentions"] = list(mentions)
        return len(mentions)

    monkeypatch.setattr(entity_extraction, "bulk_insert_entity_mentions", fake_bulk_insert)

    entity_extraction.extract_entities_batch(
        date_str="2026-05-14",
        country_config={
            "country_code": "TR",
            "country_slug": "turkey",
            "language": "tr",
            "entity_narrative": {
                "enabled": True,
                "aliases": {
                    "people": {
                        "Recep Tayyip Erdoğan": {
                            "wikidata_qid": "Q39259",
                            "aliases": ["erdoğan", "cumhurbaşkanı erdoğan"],
                        }
                    },
                    "organizations": {},
                },
            },
        },
    )

    mention = captured["mentions"][0]
    assert mention["canonical"] == "Recep Tayyip Erdoğan"
    assert mention["wikidata_qid"] == "Q39259"
    assert mention["resolver_method"] == "local_alias"
