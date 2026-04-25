"""Smoke tests for src.analysis.vector_store."""

import pytest

_FAKE_ITEMS = [
    {
        "id": 1,
        "title": "Ekonomi haberi",
        "cleaned_title": "ekonomi haberi",
        "cleaned_summary": "merkez bankası faiz kararı aldı",
        "source_name": "Hürriyet",
        "link": "https://example.com/1",
        "sentiment_label": "negative",
        "sentiment_score": 0.8,
        "cluster_id": 0,
        "date": "2026-04-20",
    },
    {
        "id": 2,
        "title": "Spor haberi",
        "cleaned_title": "spor haberi",
        "cleaned_summary": "fenerbahçe şampiyonlar ligine katıldı",
        "source_name": "NTV",
        "link": "https://example.com/2",
        "sentiment_label": "positive",
        "sentiment_score": 0.9,
        "cluster_id": 1,
        "date": "2026-04-20",
    },
]


@pytest.fixture()
def mock_db(monkeypatch):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_embed_model", None)
    monkeypatch.setattr("src.db.queries.fetch_for_indexing", lambda date_str: _FAKE_ITEMS)


def test_index_date(mock_db, tmp_path, monkeypatch):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_CHROMA_DIR", tmp_path / "chroma")

    count = vs.index_date("2026-04-20")
    assert count == 2


def test_find_similar(mock_db, tmp_path, monkeypatch):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_CHROMA_DIR", tmp_path / "chroma")

    vs.index_date("2026-04-20")

    results = vs.find_similar("faiz kararı ekonomi", n=2)
    assert len(results) >= 1
    assert "title" in results[0]
    assert "similarity" in results[0]
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_index_date_no_items(monkeypatch, tmp_path):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(vs, "_embed_model", None)
    monkeypatch.setattr("src.db.queries.fetch_for_indexing", lambda date_str: [])

    count = vs.index_date("1999-01-01")
    assert count == 0
