from __future__ import annotations


def test_pipeline_article_fetch_is_opt_in(monkeypatch):
    import src.pipeline as pipeline

    calls: list[str] = []
    monkeypatch.setattr(pipeline, "collect_all", lambda **kwargs: calls.append("collect") or 1)
    monkeypatch.setattr(pipeline, "fetch_articles", lambda **kwargs: calls.append("article_fetch") or {"parsed": 1})
    monkeypatch.setattr(pipeline, "preprocess", lambda **kwargs: calls.append("preprocess") or 1)
    monkeypatch.setattr(pipeline, "analyze", lambda **kwargs: calls.append("sentiment") or 1)
    monkeypatch.setattr(pipeline, "extract_entities", lambda **kwargs: calls.append("ner") or 1)
    monkeypatch.setattr(pipeline, "cluster_topics", lambda **kwargs: calls.append("clustering") or 1)
    monkeypatch.setattr(pipeline, "index_date", lambda **kwargs: calls.append("vector_store") or 1)
    monkeypatch.setattr(pipeline, "run_drift_check", lambda **kwargs: calls.append("drift") or {})

    pipeline.run(date_str="2026-05-05")

    assert calls == [
        "collect",
        "preprocess",
        "sentiment",
        "ner",
        "clustering",
        "vector_store",
        "drift",
    ]


def test_pipeline_article_fetch_runs_before_preprocess(monkeypatch):
    import src.pipeline as pipeline

    calls: list[str] = []
    monkeypatch.setattr(pipeline, "collect_all", lambda **kwargs: calls.append("collect") or 1)
    monkeypatch.setattr(pipeline, "fetch_articles", lambda **kwargs: calls.append("article_fetch") or {"parsed": 1})
    monkeypatch.setattr(pipeline, "preprocess", lambda **kwargs: calls.append("preprocess") or 1)
    monkeypatch.setattr(pipeline, "analyze", lambda **kwargs: calls.append("sentiment") or 1)
    monkeypatch.setattr(pipeline, "extract_entities", lambda **kwargs: calls.append("ner") or 1)
    monkeypatch.setattr(pipeline, "cluster_topics", lambda **kwargs: calls.append("clustering") or 1)
    monkeypatch.setattr(pipeline, "index_date", lambda **kwargs: calls.append("vector_store") or 1)
    monkeypatch.setattr(pipeline, "run_drift_check", lambda **kwargs: calls.append("drift") or {})

    pipeline.run(date_str="2026-05-05", fetch_article_bodies=True, article_limit=5, article_workers=2)

    assert calls[:3] == ["collect", "article_fetch", "preprocess"]
