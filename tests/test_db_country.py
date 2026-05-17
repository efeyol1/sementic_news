"""Country-aware insert path tests for ``news_items``.

Phase 4 adds ``country_code`` and ``language`` columns. The DB itself
isn't reachable from unit tests, so we monkeypatch the connection layer
and assert against the SQL fragment + params actually built by
``insert_raw_items``.
"""

from __future__ import annotations

from typing import Any

import pytest


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.execute_values_calls: list[tuple[str, list[Any]]] = []
        self._counts = iter([0, 0])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query: str, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchone(self):
        return (next(self._counts),)

    def fetchall(self):
        return []

    def executemany(self, query: str, params: Any) -> None:
        self.executed.append((query, params))


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self, **_kwargs):
        return self._cursor


@pytest.fixture
def captured_insert(monkeypatch):
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    captured: dict[str, Any] = {}

    def fake_execute_values(cur, sql, rows):
        captured["sql"] = sql
        captured["rows"] = list(rows)

    monkeypatch.setattr(queries.psycopg2.extras, "execute_values", fake_execute_values)

    yield queries, captured, cursor


def test_insert_raw_items_writes_country_code_and_language(captured_insert):
    queries, captured, cursor = captured_insert

    queries.insert_raw_items(
        [{"title": "t", "summary": "s", "source_name": "X", "link": "https://x/1"}],
        "2026-05-08",
        country_code="TR",
        language="tr",
    )

    assert "country_code, language" in captured["sql"]
    row = captured["rows"][0]
    assert row[-2] == "TR"
    assert row[-1] == "tr"
    assert row[0] == "2026-05-08"
    assert cursor.executed[0][1] == ("2026-05-08", "TR")
    assert cursor.executed[-1][1] == ("2026-05-08", "TR")


def test_insert_raw_items_defaults_to_turkey_pilot(captured_insert):
    queries, captured, _cursor = captured_insert

    queries.insert_raw_items(
        [{"title": "t", "source_name": "X", "link": "https://x/2"}],
        "2026-05-08",
    )

    row = captured["rows"][0]
    assert row[-2] == "TR"
    assert row[-1] == "tr"


def test_insert_raw_items_propagates_non_turkey_country(captured_insert):
    queries, captured, cursor = captured_insert

    queries.insert_raw_items(
        [{"title": "t", "source_name": "Y", "link": "https://y/1"}],
        "2026-05-08",
        country_code="DE",
        language="de",
    )

    row = captured["rows"][0]
    assert row[-2] == "DE"
    assert row[-1] == "de"
    assert cursor.executed[0][1] == ("2026-05-08", "DE")
    assert cursor.executed[-1][1] == ("2026-05-08", "DE")


def test_insert_raw_items_skips_when_empty(captured_insert):
    queries, captured, _cursor = captured_insert

    inserted = queries.insert_raw_items([], "2026-05-08")

    assert inserted == 0
    assert captured == {}


def test_fetch_raw_by_date_filters_by_country(monkeypatch):
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    rows = queries.fetch_raw_by_date("2026-05-08", country_code="DE")

    assert rows == []
    assert "country_code = %s" in cursor.executed[0][0]
    assert cursor.executed[0][1] == ("2026-05-08", "DE")


def test_fetch_sentiment_trend_filters_by_country(monkeypatch):
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    rows = queries.fetch_sentiment_trend(days=14, country_code="DE")

    assert rows == []
    assert "country_code = %s" in cursor.executed[0][0]
    assert cursor.executed[0][1] == ("DE", 14)


def test_upsert_cluster_summaries_is_country_scoped(monkeypatch):
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    queries.upsert_cluster_summaries(
        [
            {
                "cluster_id": 1,
                "title": "Economy",
                "size": 2,
                "keywords": ["rates"],
                "sources": {"Example": 2},
                "sentiment_distribution": {"neutral": 2},
            }
        ],
        "2026-05-08",
        country_code="DE",
    )

    sql, params = cursor.executed[0]
    assert "ON CONFLICT (country_code, date, cluster_id)" in sql
    assert params[0][0] == "DE"


# ---------------------------------------------------------------------------
# bulk_insert_entity_mentions
# ---------------------------------------------------------------------------


def _mention(article_id: int, n: int) -> dict:
    return {
        "article_id": article_id,
        "country_code": "DE",
        "collected_date": "2026-05-16",
        "entity_text": f"Entity-{article_id}-{n}",
        "entity_type": "PER",
        "position_in_article": n,
    }


def test_bulk_insert_entity_mentions_returns_total_rows_not_rowcount(monkeypatch):
    """Regression: psycopg2.extras.execute_values runs N INSERT batches
    (default ``page_size=100``); ``cur.rowcount`` only reflects the last
    batch. The helper must return ``len(rows)``, not ``cur.rowcount`` —
    otherwise the daily entity_extraction log under-reports the real
    insert volume (the 2026-05-16 DE run logged ``17`` while persisting
    ``417`` rows)."""
    import src.db.queries as queries

    cursor = _FakeCursor()
    cursor.rowcount = 17  # what execute_values' last batch would report
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))
    monkeypatch.setattr(
        queries.psycopg2.extras,
        "execute_values",
        lambda cur, sql, rows: None,
    )

    mentions = [_mention(i, j) for i in range(1, 6) for j in range(200 // 5)]
    inserted = queries.bulk_insert_entity_mentions(mentions, article_ids=[1, 2, 3, 4, 5])

    assert inserted == len(mentions) == 200


def test_bulk_insert_entity_mentions_clears_articles_when_no_mentions(monkeypatch):
    """An article that NER returned nothing for must still wipe its prior
    mentions — otherwise re-running the day would leave yesterday's
    extractions in place. Caller passes the article IDs; helper deletes
    them and returns 0 without touching execute_values."""
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    called: list[bool] = []
    monkeypatch.setattr(
        queries.psycopg2.extras,
        "execute_values",
        lambda *a, **kw: called.append(True),
    )

    inserted = queries.bulk_insert_entity_mentions([], article_ids=[10, 11, 12])

    assert inserted == 0
    assert called == []  # no INSERT path
    delete_sql, delete_params = cursor.executed[0]
    assert "DELETE FROM entity_mentions" in delete_sql
    assert delete_params == ([10, 11, 12],)


def test_bulk_insert_entity_mentions_skips_when_no_articles(monkeypatch):
    import src.db.queries as queries

    cursor = _FakeCursor()
    monkeypatch.setattr(queries, "get_conn", lambda: _FakeConnection(cursor))

    inserted = queries.bulk_insert_entity_mentions([], article_ids=[])

    assert inserted == 0
    assert cursor.executed == []
