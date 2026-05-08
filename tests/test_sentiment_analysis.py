from __future__ import annotations

import pytest


class _FakeCursor:
    def __init__(self):
        self.query = ""
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return []


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self, **_kwargs):
        return self._cursor


def test_analyze_forwards_stale_filter_and_limit(monkeypatch):
    import src.analysis.sentiment as sentiment

    captured = {}
    monkeypatch.setenv("SENTIMENT_MODEL_ID", "efeyol11/bert-turkish-sentiment")

    def fake_fetch_processed_by_date(date_str, only_missing, only_stale_after_body, only_with_body, limit):
        captured.update(
            {
                "date_str": date_str,
                "only_missing": only_missing,
                "only_stale_after_body": only_stale_after_body,
                "only_with_body": only_with_body,
                "limit": limit,
            }
        )
        return []

    monkeypatch.setattr(sentiment, "fetch_processed_by_date", fake_fetch_processed_by_date)

    result = sentiment.analyze(
        date_str="2026-05-06",
        only_stale_after_body=True,
        limit=100,
    )

    assert result == 0
    assert captured == {
        "date_str": "2026-05-06",
        "only_missing": False,
        "only_stale_after_body": True,
        "only_with_body": False,
        "limit": 100,
    }


def test_parse_args_rejects_conflicting_filters():
    import src.analysis.sentiment as sentiment

    with pytest.raises(SystemExit):
        sentiment._parse_args(["--only-missing", "--only-stale-after-body"])

    with pytest.raises(SystemExit):
        sentiment._parse_args(["--only-missing", "--only-with-body"])


def test_parse_args_rejects_non_positive_limit():
    import src.analysis.sentiment as sentiment

    with pytest.raises(SystemExit):
        sentiment._parse_args(["--limit", "0"])


def test_parse_args_rejects_scoped_write_without_production_backend(monkeypatch):
    import src.analysis.sentiment as sentiment

    monkeypatch.delenv("SENTIMENT_MODEL_ID", raising=False)
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)

    with pytest.raises(SystemExit):
        sentiment._parse_args(["--only-stale-after-body"])

    with pytest.raises(SystemExit):
        sentiment._parse_args(["--only-missing"])

    with pytest.raises(SystemExit):
        sentiment._parse_args(["--only-with-body"])


def test_parse_args_allows_stale_rescore_with_finetuned_model(monkeypatch):
    import src.analysis.sentiment as sentiment

    monkeypatch.setenv("SENTIMENT_MODEL_ID", "efeyol11/bert-turkish-sentiment")
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)

    args = sentiment._parse_args(["--only-stale-after-body", "--limit", "10"])

    assert args.only_stale_after_body is True
    assert args.limit == 10


def test_parse_args_allows_body_rescore_with_finetuned_model(monkeypatch):
    import src.analysis.sentiment as sentiment

    monkeypatch.setenv("SENTIMENT_MODEL_ID", "efeyol11/bert-turkish-sentiment")
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)

    args = sentiment._parse_args(["--only-with-body", "--limit", "10"])

    assert args.only_with_body is True
    assert args.limit == 10


def test_analyze_rejects_conflicting_filters_programmatically():
    import src.analysis.sentiment as sentiment

    with pytest.raises(ValueError, match="mutually exclusive"):
        sentiment.analyze(only_missing=True, only_stale_after_body=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        sentiment.analyze(only_missing=True, only_with_body=True)


def test_analyze_rejects_non_positive_limit_programmatically():
    import src.analysis.sentiment as sentiment

    with pytest.raises(ValueError, match="positive integer"):
        sentiment.analyze(limit=0)


def test_analyze_rejects_scoped_write_without_production_backend(monkeypatch):
    import src.analysis.sentiment as sentiment

    monkeypatch.delenv("SENTIMENT_MODEL_ID", raising=False)
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)

    with pytest.raises(ValueError, match="require SENTIMENT_MODEL_ID"):
        sentiment.analyze(only_stale_after_body=True)

    with pytest.raises(ValueError, match="require SENTIMENT_MODEL_ID"):
        sentiment.analyze(only_missing=True)

    with pytest.raises(ValueError, match="require SENTIMENT_MODEL_ID"):
        sentiment.analyze(only_with_body=True)


def test_fetch_processed_by_date_orders_body_rescore_by_oldest_analysis(monkeypatch):
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    rows = queries.fetch_processed_by_date("2026-05-06", only_with_body=True, limit=50)

    assert rows == []
    assert "LENGTH(COALESCE(cleaned_article_text, '')) > 0" in cursor.query
    assert "is_turkish = true" in cursor.query
    assert "ORDER BY analyzed_at NULLS FIRST, id" in cursor.query
    assert cursor.params == ["2026-05-06", 50]
