"""FastAPI endpoint smoke tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

_FAKE_ITEM = {
    "id": 1,
    "title": "Test Haberi",
    "source_name": "Test Kaynak",
    "published_date": "2026-04-20T10:00:00+00:00",
    "link": "https://example.com/1",
    "is_turkish": True,
    "sentiment_label": "positive",
    "sentiment_score": 0.92,
    "entities": {"PER": ["Ali"], "ORG": ["TBMM"], "LOC": ["Ankara"]},
    "cluster_id": 0,
    "cluster_title": "Test · TBMM",
    "cluster_keywords": ["test", "haber"],
}

# Sprint 7.5: non-TR fixture pinning that the API no longer post-filters
# by `is_turkish`. A German pipeline can — in the edge case where
# langdetect returns the wrong language for a short headline — produce
# rows with `is_turkish=False`. Pre-Sprint-7.5 those rows were silently
# dropped from /api/today, /api/source-comparison and /api/trend. The
# tests below assert they now flow through.
_FAKE_ITEM_DE = {
    "id": 2,
    "title": "Bundeskanzler Test",
    "source_name": "DER SPIEGEL",
    "published_date": "2026-04-20T10:00:00+00:00",
    "link": "https://example.com/de-1",
    "is_turkish": False,  # langdetect returned a non-target language
    "sentiment_label": "negative",
    "sentiment_score": 0.83,
    "entities": {"PER": ["Merkel"], "ORG": ["Bundestag"], "LOC": ["Berlin"]},
    "cluster_id": 0,
    "cluster_title": "Politik · Bundestag",
    "cluster_keywords": ["test", "politik"],
}

_FAKE_CLUSTER = {
    "cluster_id": 0,
    "title": "Test · TBMM",
    "size": 1,
    "keywords": ["test", "haber"],
    "sources": {"Test Kaynak": 1},
    "sentiment_distribution": {"positive": 1, "neutral": 0, "negative": 0},
}

# Sprint 8: entity profile fixtures (DB-row shaped). end_date is a plain
# string here — the serializer accepts date objects or strings.
_FAKE_PROFILE_ROW = {
    "country_code": "DE",
    "canonical": "Donald Trump",
    "entity_type": "PER",
    "wikidata_qid": "Q22686",
    "window_days": 30,
    "end_date": "2026-05-29",
    "coverage_days": 27,
    "total_mentions": 412,
    "total_cooccurrences": 5821,
    "window_total": 103442,
    "avg_pmi": 2.81,
    "avg_log_likelihood": 44.3,
    "top_collocates": [
        {"lemma": "zoll", "pos": "NOUN", "c11_window": 37, "pmi": 3.42, "llr": 58.1},
    ],
}

_FAKE_DIR_ROW = {
    "canonical": "Donald Trump",
    "wikidata_qid": "Q22686",
    "entity_type": "PER",
    "total_mentions": 412,
    "total_cooccurrences": 5821,
    "avg_pmi": 2.81,
    "avg_log_likelihood": 44.3,
    "coverage_days": 27,
}

_FAKE_TL_ROW = {
    "collected_date": "2026-05-29",
    "mention_count": 18,
    "total_cooccurrences": 241,
    "avg_pmi": 2.6,
    "avg_log_likelihood": 38.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FAKE_CALLS: dict[str, list[dict]] = {}


@pytest.fixture(autouse=True)
def _mock_db(monkeypatch):
    """Replace DB query functions with in-memory fakes that record
    ``country_code`` so country-aware tests can assert propagation."""
    import src.api.main as api_module

    _FAKE_CALLS.clear()

    def _record(name, **kwargs):
        _FAKE_CALLS.setdefault(name, []).append(kwargs)

    def fake_fetch_all_for_api(date_str, country_code="TR"):
        _record("fetch_all_for_api", date_str=date_str, country_code=country_code)
        return [_FAKE_ITEM]

    def fake_fetch_cluster_summaries(date_str, country_code="TR"):
        _record("fetch_cluster_summaries", date_str=date_str, country_code=country_code)
        return [_FAKE_CLUSTER]

    def fake_fetch_available_dates(country_code="TR"):
        _record("fetch_available_dates", country_code=country_code)
        return ["2026-04-20"]

    def fake_fetch_top_entities(date_str, country_code="TR", limit=10):
        _record("fetch_top_entities", date_str=date_str, country_code=country_code, limit=limit)
        return {"PER": ["Ali"], "ORG": ["TBMM"], "LOC": ["Ankara"]}

    def fake_fetch_entity_directory(
        country_code, window_days, end_date=None, q=None, entity_type=None, limit=50
    ):
        _record(
            "fetch_entity_directory", country_code=country_code,
            window_days=window_days, q=q, entity_type=entity_type, limit=limit,
        )
        return [dict(_FAKE_DIR_ROW)]

    def fake_fetch_latest_profile_end_date(country_code=None):
        _record("fetch_latest_profile_end_date", country_code=country_code)
        return "2026-05-29"

    def fake_fetch_entity_profile(
        country_code, window_days, end_date=None, qid=None, canonical=None
    ):
        _record(
            "fetch_entity_profile", country_code=country_code,
            window_days=window_days, qid=qid, canonical=canonical,
        )
        return dict(_FAKE_PROFILE_ROW)

    def fake_fetch_entity_profiles_all_countries(
        window_days, end_date=None, qid=None, canonical=None, countries=None
    ):
        _record(
            "fetch_entity_profiles_all_countries", window_days=window_days,
            qid=qid, canonical=canonical, countries=countries,
        )
        de = dict(_FAKE_PROFILE_ROW)
        fr = dict(_FAKE_PROFILE_ROW)
        fr["country_code"] = "FR"
        return [de, fr]

    def fake_fetch_entity_timeline(
        country_code, start_date, end_date, qid=None, canonical=None
    ):
        _record(
            "fetch_entity_timeline", country_code=country_code,
            start_date=start_date, end_date=end_date, qid=qid, canonical=canonical,
        )
        return [dict(_FAKE_TL_ROW)]

    monkeypatch.setattr(api_module, "fetch_all_for_api", fake_fetch_all_for_api)
    monkeypatch.setattr(api_module, "fetch_cluster_summaries", fake_fetch_cluster_summaries)
    monkeypatch.setattr(api_module, "fetch_available_dates", fake_fetch_available_dates)
    monkeypatch.setattr(api_module, "fetch_top_entities", fake_fetch_top_entities)
    monkeypatch.setattr(api_module, "fetch_entity_directory", fake_fetch_entity_directory)
    monkeypatch.setattr(
        api_module, "fetch_latest_profile_end_date", fake_fetch_latest_profile_end_date
    )
    monkeypatch.setattr(api_module, "fetch_entity_profile", fake_fetch_entity_profile)
    monkeypatch.setattr(
        api_module, "fetch_entity_profiles_all_countries",
        fake_fetch_entity_profiles_all_countries,
    )
    monkeypatch.setattr(api_module, "fetch_entity_timeline", fake_fetch_entity_timeline)
    monkeypatch.setattr(api_module, "init_db", lambda: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_today():
    r = client.get("/api/today?date=2026-04-20")
    assert r.status_code == 200
    data = r.json()
    assert data["date"] == "2026-04-20"
    assert data["total_items"] == 1
    assert data["turkish_items"] == 1
    assert "sentiment" in data
    assert "top_entities" in data


def test_today_missing_date(monkeypatch):
    import src.api.main as api_module

    monkeypatch.setattr(
        api_module, "fetch_all_for_api", lambda date_str, country_code="TR": []
    )
    r = client.get("/api/today?date=1999-01-01")
    assert r.status_code == 404


def test_topic():
    r = client.get("/api/topic/0?date=2026-04-20")
    assert r.status_code == 200
    data = r.json()
    assert data["cluster_id"] == 0
    assert len(data["news"]) == 1


def test_topic_not_found():
    r = client.get("/api/topic/5?date=2026-04-20")
    assert r.status_code == 404


def test_source_comparison():
    r = client.get("/api/source-comparison?date=2026-04-20")
    assert r.status_code == 200
    data = r.json()
    assert "Test Kaynak" in data["sources"]
    assert data["sources"]["Test Kaynak"]["total"] == 1


def test_clusters():
    r = client.get("/api/clusters?date=2026-04-20")
    assert r.status_code == 200
    data = r.json()
    assert data["date"] == "2026-04-20"
    assert data["total_clusters"] == 1
    cluster = data["clusters"][0]
    assert cluster["cluster_id"] == 0
    assert cluster["title"] == "Test · TBMM"
    assert cluster["size"] == 1
    assert cluster["keywords"] == ["test", "haber"]
    assert cluster["sources"] == {"Test Kaynak": 1}
    assert cluster["sentiment_distribution"] == {
        "positive": 1, "neutral": 0, "negative": 0,
    }


def test_clusters_empty(monkeypatch):
    import src.api.main as api_module

    monkeypatch.setattr(
        api_module, "fetch_cluster_summaries", lambda d, country_code="TR": []
    )
    r = client.get("/api/clusters?date=1999-01-01")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Country-aware endpoint tests (Phase 5)
# ---------------------------------------------------------------------------


def test_countries_endpoint_lists_active_countries():
    r = client.get("/api/countries")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert any(c["slug"] == "turkey" and c["code"] == "TR" for c in data), data
    for c in data:
        # Sprint 7.5 added timezone — must be present so the dashboard
        # can localize timestamps without hard-coding Europe/Istanbul.
        assert {"code", "slug", "name", "language", "timezone", "status"} <= c.keys()


def test_countries_endpoint_returns_country_timezone():
    """Every country YAML declares a timezone; the API must surface it."""
    r = client.get("/api/countries")
    assert r.status_code == 200
    countries = {c["slug"]: c for c in r.json()}
    assert countries["turkey"]["timezone"] == "Europe/Istanbul"
    # Sanity: at least one non-TR country surfaces its own zone.
    non_tr_zones = {
        c["slug"]: c["timezone"] for c in r.json() if c["slug"] != "turkey"
    }
    assert non_tr_zones, "expected ≥1 non-TR country with a timezone"
    assert all(z and z != "Europe/Istanbul" for z in non_tr_zones.values()), non_tr_zones


# ---------------------------------------------------------------------------
# Multi-country regression: is_turkish filter must not drop non-TR items
# (Sprint 7.5 — fixes the legacy TR-only post-filter in /api/today,
# /api/source-comparison; queries.py:fetch_sentiment_trend cleanup is
# also exercised by the dashboard's /api/trend endpoint)
# ---------------------------------------------------------------------------


def _patch_fetch_all(monkeypatch, items):
    import src.api.main as api_module
    monkeypatch.setattr(
        api_module, "fetch_all_for_api",
        lambda date_str, country_code="TR": items,
    )


def test_today_counts_non_target_language_items(monkeypatch):
    """A DE item with `is_turkish=False` must be counted in /api/today."""
    _patch_fetch_all(monkeypatch, [_FAKE_ITEM_DE])
    r = client.get("/api/today?date=2026-04-20&country=germany")
    assert r.status_code == 200
    data = r.json()
    assert data["total_items"] == 1
    # Sprint 7.5: turkish_items kept as legacy field name but now means
    # "country-pipeline items" — non-TR items count even with
    # is_turkish=False on the row.
    assert data["turkish_items"] == 1
    assert data["sentiment"]["counts"]["negative"] == 1


def test_today_top_entities_from_non_target_language_items(monkeypatch):
    """Entity aggregation must include non-`is_turkish` rows."""
    import src.api.main as api_module
    # Force the COALESCE-into-aggregation path by returning empty
    # fetch_top_entities so /api/today rebuilds entities from items.
    monkeypatch.setattr(
        api_module, "fetch_top_entities",
        lambda date_str, country_code="TR", limit=10: {"PER": [], "ORG": [], "LOC": []},
    )
    _patch_fetch_all(monkeypatch, [_FAKE_ITEM_DE])
    r = client.get("/api/today?date=2026-04-20&country=germany")
    assert r.status_code == 200
    top = r.json()["top_entities"]
    assert "Merkel" in top["PER"]
    assert "Bundestag" in top["ORG"]
    assert "Berlin" in top["LOC"]


def test_source_comparison_includes_non_target_language_items(monkeypatch):
    """A DE source must appear in /api/source-comparison even if
    is_turkish=False on every row."""
    _patch_fetch_all(monkeypatch, [_FAKE_ITEM_DE])
    r = client.get("/api/source-comparison?date=2026-04-20&country=germany")
    assert r.status_code == 200
    data = r.json()
    assert "DER SPIEGEL" in data["sources"]
    assert data["sources"]["DER SPIEGEL"]["total"] == 1
    assert data["sources"]["DER SPIEGEL"]["sentiment_counts"]["negative"] == 1


def test_trend_endpoint_propagates_country_code(monkeypatch):
    """/api/trend hits fetch_sentiment_trend with the resolved country."""
    import src.api.main as api_module
    captured: dict = {}

    def fake_trend(days, country_code="TR"):
        captured["days"] = days
        captured["country_code"] = country_code
        return [
            {"date": "2026-04-20", "positive": 3, "negative": 2,
             "neutral": 1, "total": 6},
        ]

    monkeypatch.setattr(api_module, "fetch_sentiment_trend", fake_trend)
    r = client.get("/api/trend?days=7&country=germany")
    assert r.status_code == 200
    assert captured["country_code"] == "DE"
    assert captured["days"] == 7
    assert r.json()["points"][0]["positive"] == 3


def test_today_default_country_is_turkey():
    """No ?country param → resolve_country defaults to turkey, propagates TR."""
    _FAKE_CALLS.clear()
    r = client.get("/api/today?date=2026-04-20")
    assert r.status_code == 200
    calls = _FAKE_CALLS.get("fetch_all_for_api", [])
    assert calls and calls[-1]["country_code"] == "TR"


def test_today_with_explicit_slug():
    _FAKE_CALLS.clear()
    r = client.get("/api/today?date=2026-04-20&country=turkey")
    assert r.status_code == 200
    assert _FAKE_CALLS["fetch_all_for_api"][-1]["country_code"] == "TR"


def test_today_with_iso_code():
    _FAKE_CALLS.clear()
    r = client.get("/api/today?date=2026-04-20&country=TR")
    assert r.status_code == 200
    assert _FAKE_CALLS["fetch_all_for_api"][-1]["country_code"] == "TR"


def test_today_invalid_country_returns_404():
    r = client.get("/api/today?date=2026-04-20&country=mars")
    assert r.status_code == 404
    assert "mars" in r.json()["detail"].lower()


def test_clusters_propagates_country_code():
    _FAKE_CALLS.clear()
    client.get("/api/clusters?date=2026-04-20&country=turkey")
    assert _FAKE_CALLS["fetch_cluster_summaries"][-1]["country_code"] == "TR"


def test_dates_propagates_country_code():
    _FAKE_CALLS.clear()
    client.get("/api/dates?country=turkey")
    assert _FAKE_CALLS["fetch_available_dates"][-1]["country_code"] == "TR"


# ---------------------------------------------------------------------------
# Entity profile endpoints (Sprint 8)
# ---------------------------------------------------------------------------


def test_entity_directory():
    _FAKE_CALLS.clear()
    r = client.get(
        "/api/entities?country=germany&q=trump&entity_type=PER&window_days=7&limit=5"
    )
    assert r.status_code == 200
    data = r.json()
    assert data["country_code"] == "DE"
    assert data["window_days"] == 7
    assert data["end_date"] == "2026-05-29"
    assert data["entities"][0]["canonical"] == "Donald Trump"
    call = _FAKE_CALLS["fetch_entity_directory"][-1]
    assert call["country_code"] == "DE"
    assert call["q"] == "trump"
    assert call["entity_type"] == "PER"
    assert call["window_days"] == 7
    assert call["limit"] == 5


def test_entity_directory_invalid_country():
    r = client.get("/api/entities?country=mars")
    assert r.status_code == 404


def test_entity_directory_invalid_window():
    r = client.get("/api/entities?country=germany&window_days=14")
    assert r.status_code == 400


def test_entity_profile_by_qid():
    _FAKE_CALLS.clear()
    r = client.get("/api/entity/Q22686/profile?country=germany")
    assert r.status_code == 200
    data = r.json()
    assert data["wikidata_qid"] == "Q22686"
    assert data["country_code"] == "DE"
    assert data["top_collocates"][0]["lemma"] == "zoll"
    call = _FAKE_CALLS["fetch_entity_profile"][-1]
    assert call["qid"] == "Q22686"
    assert call["canonical"] is None
    assert call["country_code"] == "DE"


def test_entity_profile_by_canonical():
    """A non-QID ref falls into the canonical branch."""
    _FAKE_CALLS.clear()
    r = client.get("/api/entity/Bundestag/profile?country=germany")
    assert r.status_code == 200
    call = _FAKE_CALLS["fetch_entity_profile"][-1]
    assert call["canonical"] == "Bundestag"
    assert call["qid"] is None


def test_entity_profile_window_param():
    _FAKE_CALLS.clear()
    client.get("/api/entity/Q22686/profile?country=germany&window_days=7")
    assert _FAKE_CALLS["fetch_entity_profile"][-1]["window_days"] == 7


def test_entity_profile_not_found(monkeypatch):
    import src.api.main as api_module
    monkeypatch.setattr(
        api_module, "fetch_entity_profile", lambda *a, **k: None
    )
    r = client.get("/api/entity/Q999999/profile?country=germany")
    assert r.status_code == 404


def test_entity_profile_invalid_country():
    r = client.get("/api/entity/Q22686/profile?country=mars")
    assert r.status_code == 404


def test_entity_compare_all_countries():
    _FAKE_CALLS.clear()
    r = client.get("/api/entity/Q22686/compare")
    assert r.status_code == 200
    data = r.json()
    assert data["reference"] == "Q22686"
    assert {c["country_code"] for c in data["countries"]} == {"DE", "FR"}
    # No ?countries → resolver gets None (all countries).
    assert _FAKE_CALLS["fetch_entity_profiles_all_countries"][-1]["countries"] is None
    assert _FAKE_CALLS["fetch_entity_profiles_all_countries"][-1]["qid"] == "Q22686"


def test_entity_compare_country_filter():
    _FAKE_CALLS.clear()
    r = client.get("/api/entity/Q22686/compare?countries=de,FR")
    assert r.status_code == 200
    assert _FAKE_CALLS["fetch_entity_profiles_all_countries"][-1]["countries"] == ["DE", "FR"]


def test_entity_compare_invalid_country():
    r = client.get("/api/entity/Q22686/compare?countries=DE,MARS")
    assert r.status_code == 400
    assert "MARS" in r.json()["detail"]


def test_entity_compare_not_found(monkeypatch):
    import src.api.main as api_module
    monkeypatch.setattr(
        api_module, "fetch_entity_profiles_all_countries", lambda *a, **k: []
    )
    r = client.get("/api/entity/Q999999/compare")
    assert r.status_code == 404


def test_entity_timeline():
    _FAKE_CALLS.clear()
    r = client.get("/api/entity/Q22686/timeline?country=germany&days=14")
    assert r.status_code == 200
    data = r.json()
    assert data["reference"] == "Q22686"
    assert data["country_code"] == "DE"
    pt = data["points"][0]
    assert pt["date"] == "2026-05-29"
    assert pt["mention_count"] == 18
    assert pt["total_cooccurrences"] == 241
    call = _FAKE_CALLS["fetch_entity_timeline"][-1]
    assert call["country_code"] == "DE"
    assert call["qid"] == "Q22686"


def test_entity_timeline_canonical_default_country():
    _FAKE_CALLS.clear()
    r = client.get("/api/entity/Bundestag/timeline")
    assert r.status_code == 200
    call = _FAKE_CALLS["fetch_entity_timeline"][-1]
    assert call["canonical"] == "Bundestag"
    assert call["qid"] is None
    assert call["country_code"] == "TR"  # default country
