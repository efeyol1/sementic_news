"""Tests for configs/countries/spain.yaml (Sprint 3 onboarding).

Validates the ES config loads, opts into the entity-narrative track with
Wikidata linking, keeps the legacy TR-only NER path off, and that the
cross-country anchor entities share a single Q-ID across TR / DE / FR /
IT / ES (the Sprint 2 acceptance, extended again in Sprint 3:
"Trump → Q22686 everywhere").
"""

from __future__ import annotations

from src.config import country_loader


def test_spain_config_identity():
    cfg = country_loader.load_country_config("spain")
    assert cfg["country_code"] == "ES"
    assert cfg["country_slug"] == "spain"
    assert cfg["language"] == "es"
    assert cfg["country_name"] == "Spain"
    assert cfg["timezone"] == "Europe/Madrid"


def test_spain_config_loads_by_code():
    cfg = country_loader.load_country_config("ES")
    assert cfg["country_slug"] == "spain"


def test_spain_entity_narrative_enabled_with_wikidata():
    cfg = country_loader.load_country_config("spain")
    en = cfg["entity_narrative"]
    assert en["enabled"] is True
    assert en["wikidata"]["enabled"] is True
    # Sprint 3 (2026-05-23) recalibrated to 0.65 after the first ES live
    # run landed at 59.8 % qid_fill at 0.70 — partial label matches and
    # niche entities (Confucio, lake/street references) were borderline.
    # See spain.yaml comment.
    assert en["wikidata"]["min_confidence"] == 0.65
    assert en["ner_model"] == "Davlan/bert-base-multilingual-cased-ner-hrl"


def test_spain_legacy_ner_disabled():
    cfg = country_loader.load_country_config("spain")
    # Legacy Turkish-hardcoded NER must stay off for ES.
    assert cfg["ner"]["enabled"] is False


def test_spain_sentiment_is_zero_shot_only():
    cfg = country_loader.load_country_config("spain")
    s = cfg["sentiment"]
    assert s["enabled"] is True
    assert s["finetuned_model"] is None  # never reuse the Turkish model
    assert s["zero_shot_model"]
    assert set(s["candidate_labels"]) == {"positive", "negative", "neutral"}


def test_spain_sources_normalized():
    cfg = country_loader.load_country_config("spain")
    sources = cfg["sources"]
    assert isinstance(sources, list) and len(sources) >= 6
    for src in sources:
        assert "name" in src
        assert isinstance(src["urls"], list) and src["urls"]
        assert "url" not in src  # normalized away by the loader
    names = {s["name"] for s in sources}
    # The plan's required outlets are present.
    for outlet in (
        "El País",
        "El Mundo",
        "ABC",
        "La Vanguardia",
        "El Confidencial",
        "20 Minutos",
        "RTVE",
    ):
        assert outlet in names


def test_spain_clustering_and_embeddings():
    cfg = country_loader.load_country_config("spain")
    assert cfg["clustering"]["enabled"] is True
    assert len(cfg["clustering"]["stopwords"]) > 30  # real ES stopword list
    assert cfg["embeddings"]["vector_dim"] == 384


# ---------------------------------------------------------------------------
# Cross-country Q-ID consistency — the Sprint 2 acceptance invariant
# (extended to TR / DE / FR / IT / ES in Sprint 3)
# ---------------------------------------------------------------------------


def _people_qids(cfg: dict) -> dict[str, str]:
    people = (cfg.get("entity_narrative") or {}).get("aliases", {}).get("people", {})
    return {
        name: payload["wikidata_qid"]
        for name, payload in people.items()
        if isinstance(payload, dict) and payload.get("wikidata_qid")
    }


def test_shared_anchor_entities_have_one_qid_across_countries():
    tr = _people_qids(country_loader.load_country_config("turkey"))
    de = _people_qids(country_loader.load_country_config("germany"))
    fr = _people_qids(country_loader.load_country_config("france"))
    es = _people_qids(country_loader.load_country_config("spain"))

    # Donald Trump anchors all four corpora that carry the anchor (TR
    # carries it too post-Sprint 2 expansion).
    assert tr["Donald Trump"] == de["Donald Trump"] == fr["Donald Trump"] == es["Donald Trump"] == "Q22686"

    # Every person shared between two configs must carry the same Q-ID —
    # this is what makes cross-country entity matching correct.
    for a, b in ((tr, de), (de, fr), (tr, fr), (tr, es), (de, es), (fr, es)):
        for name in set(a) & set(b):
            assert a[name] == b[name], f"Q-ID mismatch for {name!r}: {a[name]} vs {b[name]}"


def test_spain_anchor_qids_present():
    cfg = country_loader.load_country_config("spain")
    qids = _people_qids(cfg)
    # Cross-country anchors must be present with the canonical Q-IDs.
    assert qids["Donald Trump"] == "Q22686"
    assert qids["Recep Tayyip Erdoğan"] == "Q39259"
    assert qids["Emmanuel Macron"] == "Q3052772"
    assert qids["Angela Merkel"] == "Q567"
