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

_FAKE_CLUSTER = {
    "cluster_id": 0,
    "title": "Test · TBMM",
    "size": 1,
    "keywords": ["test", "haber"],
    "sources": {"Test Kaynak": 1},
    "sentiment_distribution": {"positive": 1, "neutral": 0, "negative": 0},
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

    monkeypatch.setattr(api_module, "fetch_all_for_api", fake_fetch_all_for_api)
    monkeypatch.setattr(api_module, "fetch_cluster_summaries", fake_fetch_cluster_summaries)
    monkeypatch.setattr(api_module, "fetch_available_dates", fake_fetch_available_dates)
    monkeypatch.setattr(api_module, "fetch_top_entities", fake_fetch_top_entities)
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
        assert {"code", "slug", "name", "language", "status"} <= c.keys()


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
