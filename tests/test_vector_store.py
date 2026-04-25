"""Smoke tests for src.analysis.vector_store (pgvector backend)."""

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

_FAKE_SIMILAR = [
    {
        "title": "Ekonomi haberi",
        "source_name": "Hürriyet",
        "date": "2026-04-20",
        "sentiment_label": "negative",
        "link": "https://example.com/1",
        "similarity": 0.91,
    }
]


class _FakeModel:
    def encode(self, texts, **kwargs):
        import numpy as np
        return np.zeros((len(texts), 384), dtype="float32")


@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "fetch_for_indexing", lambda date_str: _FAKE_ITEMS)
    monkeypatch.setattr(vs, "bulk_update_embeddings", lambda updates: None)
    monkeypatch.setattr(vs, "find_similar_pgvector", lambda emb, n=5: _FAKE_SIMILAR[:n])
    monkeypatch.setattr(vs, "_get_embed_model", lambda: _FakeModel())


def test_index_date():
    import src.analysis.vector_store as vs
    count = vs.index_date("2026-04-20")
    assert count == 2


def test_find_similar():
    import src.analysis.vector_store as vs
    results = vs.find_similar("faiz kararı ekonomi", n=1)
    assert len(results) == 1
    assert "title" in results[0]
    assert "similarity" in results[0]
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_index_date_no_items(monkeypatch):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "fetch_for_indexing", lambda date_str: [])
    count = vs.index_date("1999-01-01")
    assert count == 0
