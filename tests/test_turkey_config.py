"""Tests for configs/countries/turkey.yaml (Sprint 4 entity-narrative activation).

Validates the TR config loads, flips on the entity-narrative track with
Wikidata linking (matching IT/ES Sprint 3 calibration at 0.65), keeps the
legacy Turkish-only NER path ENABLED in parallel (TR baseline preservation
gate — sentiment / clustering / top-entities dashboard must remain
bit-identical), and that cross-country anchor entities share a single
Q-ID across TR / DE / FR (Sprint 2 invariant).
"""

from __future__ import annotations

from src.config import country_loader


def test_turkey_config_identity():
    cfg = country_loader.load_country_config("turkey")
    assert cfg["country_code"] == "TR"
    assert cfg["country_slug"] == "turkey"
    assert cfg["language"] == "tr"
    assert cfg["country_name"] == "Turkey"
    assert cfg["timezone"] == "Europe/Istanbul"


def test_turkey_config_loads_by_code():
    cfg = country_loader.load_country_config("TR")
    assert cfg["country_slug"] == "turkey"


def test_turkey_legacy_ner_stays_enabled():
    # Sprint 4 baseline preservation: legacy `ner` path must stay ON for TR
    # so the existing top-entities dashboard / news_items.entities column
    # keeps being populated bit-identically. entity_narrative runs in
    # parallel into the new entity_mentions table.
    cfg = country_loader.load_country_config("turkey")
    assert cfg["ner"]["enabled"] is True
    assert cfg["ner"]["model"] == "savasy/bert-base-turkish-ner-cased"


def test_turkey_entity_narrative_enabled_with_wikidata():
    cfg = country_loader.load_country_config("turkey")
    en = cfg["entity_narrative"]
    # Sprint 4 (2026-05-25) activation: entity-narrative track ON for TR.
    assert en["enabled"] is True
    assert en["wikidata"]["enabled"] is True
    # TR matches IT/ES Sprint 3 calibration after Sprint 4 — partial-label
    # and disambig stem behavior is similar in the multilingual NER + Wikidata
    # uselang=tr setup, so 0.65 is the right band (vs the legacy 0.85 default).
    assert en["wikidata"]["min_confidence"] == 0.65
    assert en["ner_model"] == "Davlan/bert-base-multilingual-cased-ner-hrl"


# ---------------------------------------------------------------------------
# Cross-country Q-ID consistency — Sprint 2 acceptance invariant
# ---------------------------------------------------------------------------


def _people_qids(cfg: dict) -> dict[str, str]:
    people = (cfg.get("entity_narrative") or {}).get("aliases", {}).get("people", {})
    return {
        name: payload["wikidata_qid"]
        for name, payload in people.items()
        if isinstance(payload, dict) and payload.get("wikidata_qid")
    }


def test_turkey_anchor_qids_match_other_countries():
    tr = _people_qids(country_loader.load_country_config("turkey"))
    de = _people_qids(country_loader.load_country_config("germany"))
    fr = _people_qids(country_loader.load_country_config("france"))

    # Sprint 2 acceptance: Trump anchors every corpus on the same Q-ID.
    assert tr["Donald Trump"] == "Q22686"
    assert tr["Donald Trump"] == de["Donald Trump"] == fr["Donald Trump"]

    # Erdoğan is anchored in TR (added historically) and must match any
    # other corpus that also carries him.
    assert tr["Recep Tayyip Erdoğan"] == "Q39259"

    # Every person shared between TR and another config carries the same Q-ID.
    for other in (de, fr):
        for name in set(tr) & set(other):
            assert tr[name] == other[name], (
                f"Q-ID mismatch for {name!r}: TR={tr[name]} vs other={other[name]}"
            )
