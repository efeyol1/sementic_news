"""PostgreSQL connection pool for the Semantic News pipeline and API."""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Add it to your .env file: DATABASE_URL=postgresql://user:pass@host:5432/dbname"
            )
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=url)
        logger.debug("PostgreSQL connection pool created")
    return _pool


@contextmanager
def get_conn():
    """Context manager that checks out a connection from the pool, commits on
    success and rolls back on exception, then returns the connection."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
