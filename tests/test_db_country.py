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

    yield queries, captured


def test_insert_raw_items_writes_country_code_and_language(captured_insert):
    queries, captured = captured_insert

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


def test_insert_raw_items_defaults_to_turkey_pilot(captured_insert):
    queries, captured = captured_insert

    queries.insert_raw_items(
        [{"title": "t", "source_name": "X", "link": "https://x/2"}],
        "2026-05-08",
    )

    row = captured["rows"][0]
    assert row[-2] == "TR"
    assert row[-1] == "tr"


def test_insert_raw_items_propagates_non_turkey_country(captured_insert):
    queries, captured = captured_insert

    queries.insert_raw_items(
        [{"title": "t", "source_name": "Y", "link": "https://y/1"}],
        "2026-05-08",
        country_code="DE",
        language="de",
    )

    row = captured["rows"][0]
    assert row[-2] == "DE"
    assert row[-1] == "de"


def test_insert_raw_items_skips_when_empty(captured_insert):
    queries, captured = captured_insert

    inserted = queries.insert_raw_items([], "2026-05-08")

    assert inserted == 0
    assert captured == {}
