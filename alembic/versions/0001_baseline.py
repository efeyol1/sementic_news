"""baseline schema

Captures the schema that pre-existed Alembic adoption: ``news_items``,
``cluster_summaries``, and ``drift_reports`` plus their indexes and the
pgvector extension. All DDL uses ``IF NOT EXISTS`` so applying this
migration to an already-initialised production DB is a no-op — it just
inserts the alembic_version row to mark the revision as applied.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-04-29
"""

from __future__ import annotations

from alembic import op

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS news_items (
            id               SERIAL PRIMARY KEY,
            collected_date   DATE NOT NULL,
            title            TEXT,
            summary          TEXT,
            source_name      TEXT NOT NULL DEFAULT '',
            published_date   TIMESTAMPTZ,
            link             TEXT UNIQUE,
            category         TEXT,
            cleaned_title    TEXT,
            cleaned_summary  TEXT,
            is_turkish       BOOLEAN,
            char_count       INTEGER,
            sentiment_label  TEXT,
            sentiment_score  DOUBLE PRECISION,
            sentiment_scores JSONB,
            analyzed_at      TIMESTAMPTZ,
            entities         JSONB,
            entity_count     INTEGER,
            cluster_id       INTEGER,
            cluster_keywords JSONB,
            cluster_title    TEXT,
            embedding        vector(384)
        );

        CREATE INDEX IF NOT EXISTS idx_news_date      ON news_items(collected_date);
        CREATE INDEX IF NOT EXISTS idx_news_source    ON news_items(source_name);
        CREATE INDEX IF NOT EXISTS idx_news_cluster   ON news_items(cluster_id, collected_date);
        CREATE INDEX IF NOT EXISTS idx_news_embedding ON news_items USING hnsw (embedding vector_cosine_ops);

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

        CREATE TABLE IF NOT EXISTS drift_reports (
            date            DATE PRIMARY KEY,
            status          TEXT NOT NULL,
            psi             DOUBLE PRECISION,
            severity        TEXT,
            baseline_days   INTEGER,
            today_total     INTEGER,
            today_ratios    JSONB,
            baseline_ratios JSONB,
            per_class_delta JSONB,
            computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_drift_date ON drift_reports(date DESC);
        """
    )


def downgrade() -> None:
    # Destructive on purpose — only used in local dev resets.
    # The vector extension is left in place: other schemas might depend on it.
    op.execute(
        """
        DROP TABLE IF EXISTS drift_reports;
        DROP TABLE IF EXISTS cluster_summaries;
        DROP TABLE IF EXISTS news_items;
        """
    )
