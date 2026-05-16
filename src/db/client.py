"""PostgreSQL connection pool for the Semantic News pipeline and API."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from functools import wraps

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_pool: psycopg2.pool.ThreadedConnectionPool | None = None

_CONNECTION_LOSS_SIGNATURES = (
    "ssl syscall",
    "could not receive data from server",
    "server closed the connection",
    "connection reset by peer",
    "can't assign requested address",
)


def _is_connection_loss(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(signature in message for signature in _CONNECTION_LOSS_SIGNATURES)


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Add it to your .env file: DATABASE_URL=postgresql://user:pass@host:5432/dbname"
            )
        _pool = psycopg2.pool.ThreadedConnectionPool(
            1,
            10,
            dsn=url,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
        logger.debug("PostgreSQL connection pool created")
    return _pool


def _checkout_live_conn(pool: psycopg2.pool.ThreadedConnectionPool):
    """Pull a connection from the pool, replacing it if Neon has idle-closed it."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        logger.warning("Stale DB connection detected — replacing")
        try:
            pool.putconn(conn, close=True)
        except Exception:
            pass
        return pool.getconn()


@contextmanager
def get_conn():
    """Context manager that checks out a connection from the pool, commits on
    success and rolls back on exception, then returns the connection."""
    pool = _get_pool()
    conn = _checkout_live_conn(pool)
    closed = False
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            try:
                conn.close()
            except Exception:
                pass
            closed = True
        raise
    finally:
        try:
            pool.putconn(conn, close=closed)
        except psycopg2.pool.PoolError:
            pass


def retry_on_connection_loss(
    max_attempts: int = 2, backoff_sec: float = 1.0
):
    """Replay a DB callable once if Neon drops the SSL connection mid-op.

    Long sentiment / NER inference (10–20 min) leaves the pool's connection
    sitting idle long enough for Neon's serverless compute to scale to zero
    and tear it down server-side. ``_checkout_live_conn`` revalidates with
    ``SELECT 1`` on checkout, but the kill can also happen *during* a
    ``cur.executemany``. Wrapping the bulk write in this decorator catches
    that mid-op failure, drops the dead pool, and replays the entire
    callable against a fresh connection.

    Only wrap *idempotent* operations (UPDATE-by-id, INSERT-with-ON-CONFLICT).
    A non-idempotent INSERT replayed after partial commit would double-write.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            global _pool
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except (psycopg2.DatabaseError, psycopg2.InterfaceError) as exc:
                    if not _is_connection_loss(exc):
                        raise
                    last_exc = exc
                    if attempt >= max_attempts:
                        raise
                    short_message = " ".join(str(exc).split())[:240]
                    logger.warning(
                        f"DB connection lost in {fn.__name__} "
                        f"(attempt {attempt}/{max_attempts}, "
                        f"{exc.__class__.__name__}: {short_message}). "
                        "Recreating pool and retrying."
                    )
                    if _pool is not None:
                        try:
                            _pool.closeall()
                        except Exception:
                            pass
                        _pool = None
                    time.sleep(backoff_sec)
            assert last_exc is not None  # unreachable
            raise last_exc

        return wrapper

    return decorator
