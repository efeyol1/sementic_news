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


@pytest.fixture(autouse=True)
def _mock_db(monkeypatch):
    """Replace DB query functions with in-memory fakes."""
    import src.api.main as api_module

    monkeypatch.setattr(api_module, "fetch_all_for_api", lambda date_str: [_FAKE_ITEM])
    monkeypatch.setattr(api_module, "fetch_cluster_summaries", lambda date_str: [_FAKE_CLUSTER])
    monkeypatch.setattr(api_module, "fetch_available_dates", lambda: ["2026-04-20"])
    monkeypatch.setattr(api_module, "init_db", lambda: None)
    # Reset predictor cache so each test starts clean (relevant when one
    # test mocks _get_predictor and another expects the FileNotFoundError
    # path).
    monkeypatch.setattr(api_module, "_predictor_fn", None)


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
    from unittest.mock import patch

    import src.api.main as api_module

    with patch.object(api_module, "fetch_all_for_api", return_value=[]):
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


# ---------------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------------


def _fake_predictor(text: str) -> dict:
    # Stable deterministic shape — keeps assertions independent of model output.
    return {
        "label": "neutral",
        "score": 0.85,
        "scores": {"negative": 0.05, "neutral": 0.85, "positive": 0.10},
    }


def test_predict_happy_path(monkeypatch):
    import src.api.main as api_module

    monkeypatch.setattr(api_module, "_get_predictor", lambda: _fake_predictor)

    r = client.post("/api/predict", json={"text": "Merkez Bankası faiz oranını sabit tuttu."})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "neutral"
    assert data["score"] == 0.85
    assert data["backend"] == "onnx_int8"
    assert set(data["scores"]) == {"negative", "neutral", "positive"}


def test_predict_validation_too_short():
    r = client.post("/api/predict", json={"text": "Hi"})  # < min_length=3
    assert r.status_code == 422


def test_predict_503_when_onnx_missing(monkeypatch):
    import src.api.main as api_module

    def _raise():
        raise FileNotFoundError("ONNX model not found at /tmp/missing")

    monkeypatch.setattr(api_module, "_get_predictor", _raise)

    r = client.post("/api/predict", json={"text": "Test cümlesi."})
    assert r.status_code == 503
    assert "ONNX" in r.json()["detail"]
