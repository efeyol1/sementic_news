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
