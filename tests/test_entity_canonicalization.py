from __future__ import annotations


def _cfg():
    return {
        "country_code": "TR",
        "language": "tr",
        "entity_narrative": {
            "aliases": {
                "people": {
                    "Recep Tayyip Erdoğan": {
                        "wikidata_qid": "Q39259",
                        "aliases": [
                            "erdoğan",
                            "recep tayyip erdoğan",
                            "cumhurbaşkanı erdoğan",
                        ],
                    },
                    "Donald Trump": {
                        "wikidata_qid": "Q22686",
                        "aliases": ["trump", "donald trump"],
                    },
                },
                "organizations": {},
            }
        },
    }


def test_local_aliases_canonicalize_common_people():
    from src.analysis.entity_canonicalization import canonicalize_mention

    variants = ["Erdoğan", "Recep Tayyip Erdoğan", "Cumhurbaşkanı Erdoğan"]
    results = [canonicalize_mention(text, "PER", _cfg()) for text in variants]

    assert {result.canonical for result in results} == {"Recep Tayyip Erdoğan"}
    assert {result.wikidata_qid for result in results} == {"Q39259"}
    assert {result.resolver_method for result in results} == {"local_alias"}


def test_local_alias_trump_canonicalizes_but_distinct_names_do_not_merge():
    from src.analysis.entity_canonicalization import canonicalize_mention

    cfg = _cfg()
    assert canonicalize_mention("Trump", "PER", cfg).canonical == "Donald Trump"
    assert canonicalize_mention("Barack Obama", "PER", cfg).canonical == "Barack Obama"
    assert canonicalize_mention("Obama", "PER", cfg).canonical == "Obama"


def test_normalization_folds_diacritics_punctuation_and_titles():
    from src.analysis.entity_canonicalization import normalize_entity_text

    assert normalize_entity_text("Cumhurbaşkanı Erdoğan'ın", "tr") == "erdogan in"
    assert normalize_entity_text("KANZLERIN MERKEL", "de") == "merkel"


def test_type_agnostic_alias_fallback_overrides_ner_misclassification():
    """Sprint 4 prep (2026-05-23): the multilingual NER sometimes tags
    well-known people (Trump, Erdoğan) as ORG in headlines. When an alias
    matches the normalized text under PER, the PER mapping wins even if
    the caller passed ORG — the YAML alias bucket is authoritative.

    Quantified before fix: Trump was 8/10 ORG in DE on the 7 days ending
    2026-05-23 → Q22686 never linked despite the local alias being present.
    """
    from src.analysis.entity_canonicalization import canonicalize_mention

    cfg = _cfg()
    # NER says ORG, but "trump" is a PER alias → should still return Q22686.
    result = canonicalize_mention("Trump", "ORG", cfg)
    assert result.canonical == "Donald Trump"
    assert result.wikidata_qid == "Q22686"
    assert result.resolver_method == "local_alias"

    # Same for Erdoğan as ORG.
    result = canonicalize_mention("Erdoğan", "ORG", cfg)
    assert result.canonical == "Recep Tayyip Erdoğan"
    assert result.wikidata_qid == "Q39259"

    # A genuinely-unknown text gets no fallback — falls through to display.
    result = canonicalize_mention("Some Unknown Body", "ORG", cfg)
    assert result.wikidata_qid is None
    assert result.resolver_method == "normalized"
