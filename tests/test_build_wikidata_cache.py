from __future__ import annotations

import scripts.build_wikidata_cache as bwc
from src.analysis.entity_canonicalization import CanonicalEntity


def _cfg(wikidata_enabled: bool = True):
    return {
        "country_code": "DE",
        "language": "de",
        "entity_narrative": {
            "wikidata": {"enabled": wikidata_enabled, "min_confidence": 0.85},
            "aliases": {
                "people": {
                    "Olaf Scholz": {
                        "wikidata_qid": "Q61053",
                        "aliases": ["scholz", "olaf scholz", "bundeskanzler scholz"],
                    }
                },
                "organizations": {},
            },
        },
    }


def test_local_alias_cached_without_calling_wikidata(monkeypatch):
    monkeypatch.setattr(
        bwc,
        "fetch_top_entity_texts_for_cache",
        lambda **kw: [{"entity_text": "Scholz", "entity_type": "PER", "mention_count": 42}],
    )

    captured: dict = {}
    monkeypatch.setattr(
        bwc, "upsert_entity_resolution_cache", lambda rows: captured.update(rows=rows)
    )

    def _boom(*a, **k):
        raise AssertionError("_resolve_wikidata should not be called for local alias")

    monkeypatch.setattr(bwc, "_resolve_wikidata", _boom)

    stats = bwc.build_wikidata_cache(_cfg(), since_date="2026-05-01", sleep=0.0)

    assert stats == {"total": 1, "resolved_qid": 0, "local_alias": 1, "miss": 0}
    rows = captured["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["normalized_text"] == "scholz"
    assert row["entity_type"] == "PER"
    assert row["country_code"] == "DE"
    assert row["canonical"] == "Olaf Scholz"
    assert row["wikidata_qid"] == "Q61053"
    assert row["resolver_method"] == "local_alias"


def test_non_alias_entity_triggers_wikidata(monkeypatch):
    monkeypatch.setattr(
        bwc,
        "fetch_top_entity_texts_for_cache",
        lambda **kw: [
            {"entity_text": "Friedrich Merz", "entity_type": "PER", "mention_count": 17}
        ],
    )

    captured: dict = {}
    monkeypatch.setattr(
        bwc, "upsert_entity_resolution_cache", lambda rows: captured.update(rows=rows)
    )

    calls: list = []

    def _fake_resolve(text, entity_type, country_config, min_confidence):
        calls.append((text, entity_type, min_confidence))
        return CanonicalEntity(
            canonical="Friedrich Merz",
            wikidata_qid="Q1399491",
            confidence=0.93,
            resolver_method="wikidata",
        )

    monkeypatch.setattr(bwc, "_resolve_wikidata", _fake_resolve)

    stats = bwc.build_wikidata_cache(_cfg(), since_date="2026-05-01", sleep=0.0)

    assert calls == [("Friedrich Merz", "PER", 0.85)]
    assert stats == {"total": 1, "resolved_qid": 1, "local_alias": 0, "miss": 0}
    row = captured["rows"][0]
    assert row["normalized_text"] == "friedrich merz"
    assert row["entity_type"] == "PER"
    assert row["country_code"] == "DE"
    assert row["canonical"] == "Friedrich Merz"
    assert row["wikidata_qid"] == "Q1399491"
    assert row["confidence"] == 0.93
    assert row["resolver_method"] == "wikidata"


def test_wikidata_disabled_no_network_call(monkeypatch):
    monkeypatch.setattr(
        bwc,
        "fetch_top_entity_texts_for_cache",
        lambda **kw: [
            {"entity_text": "Friedrich Merz", "entity_type": "PER", "mention_count": 5}
        ],
    )

    captured: dict = {}
    monkeypatch.setattr(
        bwc, "upsert_entity_resolution_cache", lambda rows: captured.update(rows=rows)
    )

    def _boom(*a, **k):
        raise AssertionError("_resolve_wikidata must not run when wikidata disabled")

    monkeypatch.setattr(bwc, "_resolve_wikidata", _boom)

    stats = bwc.build_wikidata_cache(
        _cfg(wikidata_enabled=False), since_date="2026-05-01", sleep=0.0
    )

    assert stats == {"total": 1, "resolved_qid": 0, "local_alias": 0, "miss": 1}
    row = captured["rows"][0]
    # falls back to deterministic display canonical, no QID
    assert row["wikidata_qid"] is None
    assert row["resolver_method"] == "normalized"
    assert row["normalized_text"] == "friedrich merz"
