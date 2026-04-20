"""FastAPI endpoint smoke tests."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures — minimal fake data so tests never need real data/analyzed/ files
# ---------------------------------------------------------------------------

_FAKE_ITEM = {
    "title": "Test Haberi",
    "summary": "Test özeti",
    "source_name": "Test Kaynak",
    "published_date": "2026-04-20T10:00:00+00:00",
    "link": "https://example.com/1",
    "category": None,
    "cleaned_title": "test haberi",
    "cleaned_summary": "test özeti",
    "is_turkish": True,
    "char_count": 22,
    "sentiment_label": "positive",
    "sentiment_score": 0.92,
    "sentiment_scores": {"positive": 0.92, "negative": 0.08},
    "analyzed_at": "2026-04-20T10:01:00+00:00",
    "entities": {"PER": ["Ali"], "ORG": ["TBMM"], "LOC": ["Ankara"]},
    "entity_count": 3,
    "cluster_id": 0,
    "cluster_keywords": ["test", "haber"],
}

_FAKE_CLUSTER = {
    "cluster_id": 0,
    "size": 1,
    "keywords": ["test", "haber"],
    "sources": {"Test Kaynak": 1},
    "sentiment_distribution": {"positive": 1},
}


@pytest.fixture(autouse=True)
def _mock_data(tmp_path, monkeypatch):
    """Redirect _ANALYZED_DIR to a temp dir with fake JSON files."""
    analyzed_dir = tmp_path / "analyzed"
    analyzed_dir.mkdir()

    date_str = "2026-04-20"
    (analyzed_dir / f"{date_str}.json").write_text(
        json.dumps([_FAKE_ITEM]), encoding="utf-8"
    )
    (analyzed_dir / f"{date_str}_clusters.json").write_text(
        json.dumps([_FAKE_CLUSTER]), encoding="utf-8"
    )

    import src.api.main as api_module
    monkeypatch.setattr(api_module, "_ANALYZED_DIR", analyzed_dir)


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


def test_today_missing_date():
    r = client.get("/api/today?date=1999-01-01")
    assert r.status_code == 404


def test_topic():
    r = client.get("/api/topic/0?date=2026-04-20")
    assert r.status_code == 200
    data = r.json()
    assert data["cluster_id"] == 0
    assert len(data["news"]) == 1


def test_topic_not_found():
    r = client.get("/api/topic/999?date=2026-04-20")
    assert r.status_code == 404


def test_source_comparison():
    r = client.get("/api/source-comparison?date=2026-04-20")
    assert r.status_code == 200
    data = r.json()
    assert "Test Kaynak" in data["sources"]
    assert data["sources"]["Test Kaynak"]["total"] == 1
