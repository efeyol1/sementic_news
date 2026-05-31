"""API tests for /api/entity/{ref}/explain (Sprint 11). TestClient, stubbed."""

from __future__ import annotations

from fastapi.testclient import TestClient

import src.rag as rag_pkg
from src.api import main as api_main
from src.api.main import app

client = TestClient(app)

_PROFILE = {
    "country_code": "DE", "canonical": "Donald Trump", "entity_type": "PER",
    "wikidata_qid": "Q22686", "window_days": 30, "end_date": "2026-05-29",
    "total_mentions": 100, "coverage_days": 27, "avg_pmi": 2.1,
    "avg_log_likelihood": 40.0, "top_collocates": [], "frame_intensities": None,
}


def test_gate_off_returns_available_false():
    # turkey has rag.enabled: false
    r = client.get("/api/entity/Q22686/explain", params={"country": "turkey"})
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_no_profile_404(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_entity_profile", lambda *a, **k: None)
    r = client.get("/api/entity/Q22686/explain", params={"country": "germany"})
    assert r.status_code == 404


def test_returns_explanation(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_entity_profile", lambda *a, **k: _PROFILE)
    monkeypatch.setattr(
        rag_pkg, "explain_entity",
        lambda profile, cfg, lang: (
            {"backend": "extractive", "insufficient_evidence": False,
             "frame_summary": "ok", "claims": [], "citations": []},
            False,
        ),
    )
    r = client.get("/api/entity/Q22686/explain", params={"country": "germany"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["backend"] == "extractive"
    assert body["frame_summary"] == "ok"


def test_never_500_on_explain_error(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_entity_profile", lambda *a, **k: _PROFILE)

    def _boom(*a, **k):
        raise RuntimeError("explain blew up")

    monkeypatch.setattr(rag_pkg, "explain_entity", _boom)
    r = client.get("/api/entity/Q22686/explain", params={"country": "germany"})
    assert r.status_code == 200
    assert r.json()["insufficient_evidence"] is True


def test_invalid_window_400(monkeypatch):
    monkeypatch.setattr(api_main, "fetch_entity_profile", lambda *a, **k: _PROFILE)
    r = client.get(
        "/api/entity/Q22686/explain",
        params={"country": "germany", "window_days": 99},
    )
    assert r.status_code == 400
