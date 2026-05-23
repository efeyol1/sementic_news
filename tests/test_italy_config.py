"""Tests for configs/countries/italy.yaml (Sprint 3 onboarding).

Validates the IT config loads, opts into the entity-narrative track with
Wikidata linking, keeps the legacy TR-only NER path off, and that the
cross-country anchor entities share a single Q-ID across TR / DE / FR / IT
(the Sprint 2 acceptance, extended in Sprint 3: "Trump → Q22686 everywhere").
"""

from __future__ import annotations

from src.config import country_loader


def test_italy_config_identity():
    cfg = country_loader.load_country_config("italy")
    assert cfg["country_code"] == "IT"
    assert cfg["country_slug"] == "italy"
    assert cfg["language"] == "it"
    assert cfg["country_name"] == "Italy"
    assert cfg["timezone"] == "Europe/Rome"


def test_italy_config_loads_by_code():
    cfg = country_loader.load_country_config("IT")
    assert cfg["country_slug"] == "italy"


def test_italy_entity_narrative_enabled_with_wikidata():
    cfg = country_loader.load_country_config("italy")
    en = cfg["entity_narrative"]
    assert en["enabled"] is True
    assert en["wikidata"]["enabled"] is True
    # Sprint 3 (2026-05-23) recalibrated to 0.65 after the first IT live
    # run landed at 53.7 % qid_fill at 0.70 — niche entities (Federal
    # Reserve, Palazzo Montecitorio, lago Lemano) were legit Wikidata hits
    # with partial label matches scoring 0.55–0.60. See italy.yaml comment.
    assert en["wikidata"]["min_confidence"] == 0.65
    assert en["ner_model"] == "Davlan/bert-base-multilingual-cased-ner-hrl"


def test_italy_legacy_ner_disabled():
    cfg = country_loader.load_country_config("italy")
    # Legacy Turkish-hardcoded NER must stay off for IT.
    assert cfg["ner"]["enabled"] is False


def test_italy_sentiment_is_zero_shot_only():
    cfg = country_loader.load_country_config("italy")
    s = cfg["sentiment"]
    assert s["enabled"] is True
    assert s["finetuned_model"] is None  # never reuse the Turkish model
    assert s["zero_shot_model"]
    assert set(s["candidate_labels"]) == {"positive", "negative", "neutral"}


def test_italy_sources_normalized():
    cfg = country_loader.load_country_config("italy")
    sources = cfg["sources"]
    assert isinstance(sources, list) and len(sources) >= 6
    for src in sources:
        assert "name" in src
        assert isinstance(src["urls"], list) and src["urls"]
        assert "url" not in src  # normalized away by the loader
    names = {s["name"] for s in sources}
    # The plan's required outlets are present.
    for outlet in ("Corriere della Sera", "La Repubblica", "ANSA", "Il Sole 24 Ore", "Il Fatto Quotidiano"):
        assert outlet in names


def test_italy_clustering_and_embeddings():
    cfg = country_loader.load_country_config("italy")
    assert cfg["clustering"]["enabled"] is True
    assert len(cfg["clustering"]["stopwords"]) > 30  # real IT stopword list
    assert cfg["embeddings"]["vector_dim"] == 384


# ---------------------------------------------------------------------------
# Cross-country Q-ID consistency — the Sprint 2 acceptance invariant
# (extended to TR / DE / FR / IT in Sprint 3)
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
    it = _people_qids(country_loader.load_country_config("italy"))

    # Donald Trump anchors all four corpora.
    assert tr["Donald Trump"] == de["Donald Trump"] == fr["Donald Trump"] == it["Donald Trump"] == "Q22686"

    # Every person shared between two configs must carry the same Q-ID —
    # this is what makes cross-country entity matching correct.
    for a, b in ((tr, de), (de, fr), (tr, fr), (tr, it), (de, it), (fr, it)):
        for name in set(a) & set(b):
            assert a[name] == b[name], f"Q-ID mismatch for {name!r}: {a[name]} vs {b[name]}"
