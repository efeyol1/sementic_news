"""Database schema: DDL for news_items and cluster_summaries tables.

Run directly to initialize the database:
    python -m src.db.schema
"""

from loguru import logger

from src.db.client import get_conn

_DDL = """
CREATE TABLE IF NOT EXISTS news_items (
    id               SERIAL PRIMARY KEY,
    collected_date   DATE NOT NULL,
    title            TEXT,
    summary          TEXT,
    source_name      TEXT NOT NULL DEFAULT '',
    published_date   TIMESTAMPTZ,
    link             TEXT UNIQUE,
    category         TEXT,
    -- preprocessor
    cleaned_title    TEXT,
    cleaned_summary  TEXT,
    is_turkish       BOOLEAN,
    char_count       INTEGER,
    -- sentiment
    sentiment_label  TEXT,
    sentiment_score  DOUBLE PRECISION,
    sentiment_scores JSONB,
    analyzed_at      TIMESTAMPTZ,
    -- ner
    entities         JSONB,
    entity_count     INTEGER,
    -- clustering
    cluster_id       INTEGER,
    cluster_keywords JSONB,
    cluster_title    TEXT
);

CREATE INDEX IF NOT EXISTS idx_news_date    ON news_items(collected_date);
CREATE INDEX IF NOT EXISTS idx_news_source  ON news_items(source_name);
CREATE INDEX IF NOT EXISTS idx_news_cluster ON news_items(cluster_id, collected_date);

CREATE TABLE IF NOT EXISTS cluster_summaries (
    id                     SERIAL PRIMARY KEY,
    date                   DATE NOT NULL,
    cluster_id             INTEGER NOT NULL,
    title                  TEXT,
    size                   INTEGER,
    keywords               JSONB,
    sources                JSONB,
    sentiment_distribution JSONB,
    UNIQUE(date, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_date ON cluster_summaries(date);
"""


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
    logger.success("Database schema initialised")


if __name__ == "__main__":
    init_db()
