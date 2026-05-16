from __future__ import annotations

import psycopg2
import pytest


class _FakePool:
    def __init__(self) -> None:
        self.closeall_calls = 0

    def closeall(self) -> None:
        self.closeall_calls += 1


def test_retry_on_connection_loss_retries_database_error_signature(monkeypatch):
    import src.db.client as client

    fake_pool = _FakePool()
    monkeypatch.setattr(client, "_pool", fake_pool)
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)
    calls = 0

    @client.retry_on_connection_loss(max_attempts=2)
    def flaky_write():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise psycopg2.DatabaseError(
                "SSL SYSCALL error: Connection reset by peer"
            )
        return "ok"

    assert flaky_write() == "ok"
    assert calls == 2
    assert fake_pool.closeall_calls == 1
    assert client._pool is None


def test_retry_on_connection_loss_does_not_retry_non_connection_database_error(monkeypatch):
    import src.db.client as client

    fake_pool = _FakePool()
    monkeypatch.setattr(client, "_pool", fake_pool)
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)
    calls = 0

    @client.retry_on_connection_loss(max_attempts=2)
    def invalid_write():
        nonlocal calls
        calls += 1
        raise psycopg2.DatabaseError(
            "duplicate key value violates unique constraint"
        )

    with pytest.raises(psycopg2.DatabaseError):
        invalid_write()

    assert calls == 1
    assert fake_pool.closeall_calls == 0


def test_bulk_update_preprocessed_is_retried_by_decorator():
    import src.db.queries as queries

    assert hasattr(queries.bulk_update_preprocessed, "__wrapped__")
