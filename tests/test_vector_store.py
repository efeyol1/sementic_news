"""Smoke tests for src.analysis.vector_store."""

import json
import pytest


_FAKE_ITEMS = [
    {
        "title": "Ekonomi haberi",
        "cleaned_title": "ekonomi haberi",
        "cleaned_summary": "merkez bankası faiz kararı aldı",
        "is_turkish": True,
        "sentiment_label": "negative",
        "sentiment_score": 0.8,
        "source_name": "Hürriyet",
        "link": "https://example.com/1",
        "cluster_id": 0,
    },
    {
        "title": "Spor haberi",
        "cleaned_title": "spor haberi",
        "cleaned_summary": "fenerbahçe şampiyonlar ligine katıldı",
        "is_turkish": True,
        "sentiment_label": "positive",
        "sentiment_score": 0.9,
        "source_name": "NTV",
        "link": "https://example.com/2",
        "cluster_id": 1,
    },
]


@pytest.fixture()
def fake_analyzed(tmp_path):
    d = tmp_path / "analyzed"
    d.mkdir()
    (d / "2026-04-20.json").write_text(json.dumps(_FAKE_ITEMS), encoding="utf-8")
    return d


def test_index_date(fake_analyzed, monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma"

    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_DATA_ANALYZED_DIR", fake_analyzed)
    monkeypatch.setattr(vs, "_CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(vs, "_embed_model", None)

    count = vs.index_date("2026-04-20")
    assert count == 2


def test_find_similar(fake_analyzed, monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma"

    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_DATA_ANALYZED_DIR", fake_analyzed)
    monkeypatch.setattr(vs, "_CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(vs, "_embed_model", None)

    vs.index_date("2026-04-20")

    results = vs.find_similar("faiz kararı ekonomi", n=2)
    assert len(results) >= 1
    assert "title" in results[0]
    assert "similarity" in results[0]
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_index_date_missing_file(fake_analyzed, monkeypatch, tmp_path):
    import src.analysis.vector_store as vs
    monkeypatch.setattr(vs, "_DATA_ANALYZED_DIR", fake_analyzed)
    monkeypatch.setattr(vs, "_CHROMA_DIR", tmp_path / "chroma")

    with pytest.raises(FileNotFoundError):
        vs.index_date("1999-01-01")
