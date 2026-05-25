"""Tests for configs/countries/uk.yaml (Sprint 4 onboarding).

Validates the GB config loads, opts into the entity-narrative track with
Wikidata linking, keeps the legacy TR-only NER path off, and that the
cross-country anchor entities share a single Q-ID across
TR / DE / FR / IT / ES / GB (the Sprint 2 acceptance, extended in
Sprint 4: "Trump → Q22686 everywhere").
"""

from __future__ import annotations

from src.config import country_loader


def test_uk_config_identity():
    cfg = country_loader.load_country_config("uk")
    assert cfg["country_code"] == "GB"
    assert cfg["country_slug"] == "uk"
    assert cfg["language"] == "en"
    assert cfg["country_name"] == "United Kingdom"
    assert cfg["timezone"] == "Europe/London"


def test_uk_config_loads_by_code():
    cfg = country_loader.load_country_config("GB")
    assert cfg["country_slug"] == "uk"


def test_uk_entity_narrative_enabled_with_wikidata():
    cfg = country_loader.load_country_config("uk")
    en = cfg["entity_narrative"]
    assert en["enabled"] is True
    assert en["wikidata"]["enabled"] is True
    # Sprint 4 EN keeps 0.70 vs IT/ES 0.65 — Wikidata English
    # descriptions are richer than any other language, so partial label
    # matches already score above 0.70. Lowering further would admit
    # surname-disambig false positives (e.g. "Johnson" → Boris vs LBJ).
    # See uk.yaml comment.
    assert en["wikidata"]["min_confidence"] == 0.70
    assert en["ner_model"] == "Davlan/bert-base-multilingual-cased-ner-hrl"


def test_uk_legacy_ner_disabled():
    cfg = country_loader.load_country_config("uk")
    # Legacy Turkish-hardcoded NER must stay off for GB.
    assert cfg["ner"]["enabled"] is False


def test_uk_sentiment_is_zero_shot_only():
    cfg = country_loader.load_country_config("uk")
    s = cfg["sentiment"]
    assert s["enabled"] is True
    assert s["finetuned_model"] is None  # never reuse the Turkish model
    assert s["zero_shot_model"]
    assert set(s["candidate_labels"]) == {"positive", "negative", "neutral"}


def test_uk_sources_normalized():
    cfg = country_loader.load_country_config("uk")
    sources = cfg["sources"]
    assert isinstance(sources, list) and len(sources) >= 6
    for src in sources:
        assert "name" in src
        assert isinstance(src["urls"], list) and src["urls"]
        assert "url" not in src  # normalized away by the loader
    names = {s["name"] for s in sources}
    # The plan's required outlets are present.
    for outlet in ("BBC News", "The Guardian", "Sky News", "The Independent", "Channel 4 News", "Financial Times"):
        assert outlet in names


def test_uk_clustering_and_embeddings():
    cfg = country_loader.load_country_config("uk")
    assert cfg["clustering"]["enabled"] is True
    assert len(cfg["clustering"]["stopwords"]) > 30  # real EN stopword list
    assert cfg["embeddings"]["vector_dim"] == 384


# ---------------------------------------------------------------------------
# Cross-country Q-ID consistency — the Sprint 2 acceptance invariant
# (extended to TR / DE / FR / IT / ES / GB in Sprint 4)
# ---------------------------------------------------------------------------


def _people_qids(cfg: dict) -> dict[str, str]:
    people = (cfg.get("entity_narrative") or {}).get("aliases", {}).get("people", {})
    return {
        name: payload["wikidata_qid"]
        for name, payload in people.items()
        if isinstance(payload, dict) and payload.get("wikidata_qid")
    }


def test_uk_trump_qid_anchor():
    # Sprint 2 acceptance invariant, narrowed to UK: Donald Trump must
    # resolve to Q22686 in the UK config too.
    cfg = country_loader.load_country_config("uk")
    qids = _people_qids(cfg)
    assert qids["Donald Trump"] == "Q22686"


def test_shared_anchor_entities_have_one_qid_across_countries():
    tr = _people_qids(country_loader.load_country_config("turkey"))
    de = _people_qids(country_loader.load_country_config("germany"))
    fr = _people_qids(country_loader.load_country_config("france"))
    it = _people_qids(country_loader.load_country_config("italy"))
    es = _people_qids(country_loader.load_country_config("spain"))
    gb = _people_qids(country_loader.load_country_config("uk"))

    # Donald Trump anchors all six corpora.
    assert (
        tr["Donald Trump"]
        == de["Donald Trump"]
        == fr["Donald Trump"]
        == it["Donald Trump"]
        == es["Donald Trump"]
        == gb["Donald Trump"]
        == "Q22686"
    )

    # Every person shared between two configs must carry the same Q-ID —
    # this is what makes cross-country entity matching correct.
    configs = {"tr": tr, "de": de, "fr": fr, "it": it, "es": es, "gb": gb}
    items = list(configs.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_name, a = items[i]
            b_name, b = items[j]
            for name in set(a) & set(b):
                assert a[name] == b[name], (
                    f"Q-ID mismatch for {name!r} between {a_name} and {b_name}: "
                    f"{a[name]} vs {b[name]}"
                )
