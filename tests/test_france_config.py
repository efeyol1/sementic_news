"""Tests for configs/countries/france.yaml (Sprint 2 onboarding).

Validates the FR config loads, opts into the entity-narrative track with
Wikidata linking, keeps the legacy TR-only NER path off, and that the
cross-country anchor entities share a single Q-ID across TR / DE / FR
(the Sprint 2 acceptance: "Trump → Q22686 everywhere").
"""

from __future__ import annotations

from src.config import country_loader


def test_france_config_identity():
    cfg = country_loader.load_country_config("france")
    assert cfg["country_code"] == "FR"
    assert cfg["country_slug"] == "france"
    assert cfg["language"] == "fr"
    assert cfg["country_name"] == "France"


def test_france_config_loads_by_code():
    cfg = country_loader.load_country_config("FR")
    assert cfg["country_slug"] == "france"


def test_france_entity_narrative_enabled_with_wikidata():
    cfg = country_loader.load_country_config("france")
    en = cfg["entity_narrative"]
    assert en["enabled"] is True
    assert en["wikidata"]["enabled"] is True
    # Calibrated to 0.70 on 2026-05-20 after multilingual scoring fix; see
    # comment in configs/countries/france.yaml. Stays above 0.5 to keep
    # ambiguous single-token mentions ("Patrick", "Trump" as keyword) out.
    assert en["wikidata"]["min_confidence"] == 0.70
    assert en["ner_model"] == "Davlan/bert-base-multilingual-cased-ner-hrl"


def test_france_legacy_ner_disabled():
    cfg = country_loader.load_country_config("france")
    # Legacy Turkish-hardcoded NER must stay off for FR.
    assert cfg["ner"]["enabled"] is False


def test_france_sentiment_is_zero_shot_only():
    cfg = country_loader.load_country_config("france")
    s = cfg["sentiment"]
    assert s["enabled"] is True
    assert s["finetuned_model"] is None  # never reuse the Turkish model
    assert s["zero_shot_model"]
    assert set(s["candidate_labels"]) == {"positive", "negative", "neutral"}


def test_france_sources_normalized():
    cfg = country_loader.load_country_config("france")
    sources = cfg["sources"]
    assert isinstance(sources, list) and len(sources) >= 6
    for src in sources:
        assert "name" in src
        assert isinstance(src["urls"], list) and src["urls"]
        assert "url" not in src  # normalized away by the loader
    names = {s["name"] for s in sources}
    # The plan's required outlets are present.
    for outlet in ("Le Monde", "Le Figaro", "Libération", "Le Parisien", "France 24"):
        assert outlet in names


def test_france_clustering_and_embeddings():
    cfg = country_loader.load_country_config("france")
    assert cfg["clustering"]["enabled"] is True
    assert len(cfg["clustering"]["stopwords"]) > 30  # real FR stopword list
    assert cfg["embeddings"]["vector_dim"] == 384


# ---------------------------------------------------------------------------
# Cross-country Q-ID consistency — the Sprint 2 acceptance invariant
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

    # Donald Trump anchors all three corpora.
    assert tr["Donald Trump"] == de["Donald Trump"] == fr["Donald Trump"] == "Q22686"

    # Every person shared between two configs must carry the same Q-ID —
    # this is what makes cross-country entity matching correct.
    for a, b in ((tr, de), (de, fr), (tr, fr)):
        for name in set(a) & set(b):
            assert a[name] == b[name], f"Q-ID mismatch for {name!r}: {a[name]} vs {b[name]}"
