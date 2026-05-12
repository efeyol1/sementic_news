from __future__ import annotations


def test_pipeline_article_fetch_is_opt_in(monkeypatch):
    import src.pipeline as pipeline

    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "load_country_config",
        lambda country: {"country_code": "TR", "country_slug": "turkey", "language": "tr"},
    )
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
    monkeypatch.setattr(
        pipeline,
        "load_country_config",
        lambda country: {"country_code": "TR", "country_slug": "turkey", "language": "tr"},
    )
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


def test_pipeline_forwards_country_config_to_steps(monkeypatch):
    import src.pipeline as pipeline

    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        pipeline,
        "load_country_config",
        lambda country: {"country_code": "DE", "country_slug": "germany", "language": "de"},
    )

    def capture(name):
        return lambda **kwargs: captured.setdefault(name, kwargs) or 1

    monkeypatch.setattr(pipeline, "collect_all", capture("collect"))
    monkeypatch.setattr(pipeline, "fetch_articles", capture("article_fetch"))
    monkeypatch.setattr(pipeline, "preprocess", capture("preprocess"))
    monkeypatch.setattr(pipeline, "analyze", capture("sentiment"))
    monkeypatch.setattr(pipeline, "extract_entities", capture("ner"))
    monkeypatch.setattr(pipeline, "cluster_topics", capture("clustering"))
    monkeypatch.setattr(pipeline, "index_date", capture("vector_store"))
    monkeypatch.setattr(pipeline, "run_drift_check", capture("drift"))

    pipeline.run(
        date_str="2026-05-08",
        country="germany",
        fetch_article_bodies=True,
        article_limit=7,
        article_workers=3,
    )

    assert captured["collect"]["country"] == "germany"
    assert captured["article_fetch"]["country_code"] == "DE"
    assert captured["article_fetch"]["country_slug"] == "germany"
    assert captured["preprocess"]["country_code"] == "DE"
    assert captured["preprocess"]["target_language"] == "de"
    assert captured["sentiment"]["country_code"] == "DE"
    assert captured["sentiment"]["lang"] == "de"
    assert captured["ner"]["country_code"] == "DE"
    assert captured["clustering"]["country_code"] == "DE"
    assert captured["vector_store"]["country_code"] == "DE"
    assert captured["drift"]["country_code"] == "DE"


def test_parse_args_accepts_country():
    import src.pipeline as pipeline

    args = pipeline._parse_args(["--country", "TR", "--date", "2026-05-08"])

    assert args.country == "TR"
    assert args.date == "2026-05-08"


def test_pipeline_skips_disabled_ner(monkeypatch):
    import src.pipeline as pipeline

    calls: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "load_country_config",
        lambda country: {
            "country_code": "DE",
            "country_slug": "germany",
            "language": "de",
            "ner": {"enabled": False},
            "sentiment": {"enabled": True},
            "clustering": {"enabled": True},
            "embeddings": {"enabled": True},
        },
    )
    monkeypatch.setattr(pipeline, "collect_all", lambda **kwargs: calls.append("collect") or 1)
    monkeypatch.setattr(pipeline, "preprocess", lambda **kwargs: calls.append("preprocess") or 1)
    monkeypatch.setattr(pipeline, "analyze", lambda **kwargs: calls.append("sentiment") or 1)
    monkeypatch.setattr(pipeline, "extract_entities", lambda **kwargs: calls.append("ner") or 1)
    monkeypatch.setattr(pipeline, "cluster_topics", lambda **kwargs: calls.append("clustering") or 1)
    monkeypatch.setattr(pipeline, "index_date", lambda **kwargs: calls.append("vector_store") or 1)
    monkeypatch.setattr(pipeline, "run_drift_check", lambda **kwargs: calls.append("drift") or {})

    pipeline.run(date_str="2026-05-08", country="germany")

    assert "ner" not in calls
    assert calls == ["collect", "preprocess", "sentiment", "clustering", "vector_store", "drift"]


def test_pipeline_rejects_finetuned_sentiment_without_country_model(monkeypatch):
    import pytest

    import src.pipeline as pipeline

    monkeypatch.setenv("SENTIMENT_MODEL_ID", "efeyol11/bert-turkish-sentiment")
    monkeypatch.setattr(
        pipeline,
        "load_country_config",
        lambda country: {
            "country_code": "DE",
            "country_slug": "germany",
            "language": "de",
            "sentiment": {"enabled": True, "finetuned_model": None},
        },
    )
    monkeypatch.setattr(pipeline, "collect_all", lambda **kwargs: 1)
    monkeypatch.setattr(pipeline, "preprocess", lambda **kwargs: 1)

    with pytest.raises(RuntimeError, match="does not declare sentiment.finetuned_model"):
        pipeline.run(date_str="2026-05-08", country="germany")
