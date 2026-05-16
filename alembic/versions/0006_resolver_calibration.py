"""entity resolution cache and sentiment calibration fields

Revision ID: 0006_resolver_calibration
Revises: 0005_entity_mentions
Create Date: 2026-05-16
"""

from __future__ import annotations

from alembic import op

revision: str = "0006_resolver_calibration"
down_revision: str | None = "0005_entity_mentions"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE news_items
            ADD COLUMN IF NOT EXISTS raw_sentiment_score DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS calibrated_sentiment_score DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS calibrated_sentiment_scores JSONB,
            ADD COLUMN IF NOT EXISTS calibration_method TEXT;

        UPDATE news_items
        SET raw_sentiment_score = sentiment_score
        WHERE raw_sentiment_score IS NULL
          AND sentiment_score IS NOT NULL;

        ALTER TABLE entity_mentions
            ADD COLUMN IF NOT EXISTS resolution_confidence DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS resolver_method TEXT;

        CREATE TABLE IF NOT EXISTS entity_resolution_cache (
            normalized_text TEXT NOT NULL,
            entity_type     VARCHAR(8) NOT NULL,
            country_code    VARCHAR(2) NOT NULL,
            canonical       TEXT,
            wikidata_qid    VARCHAR(16),
            confidence      DOUBLE PRECISION,
            resolver_method TEXT,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (normalized_text, entity_type, country_code)
        );

        CREATE INDEX IF NOT EXISTS idx_entity_resolution_cache_qid
            ON entity_resolution_cache(wikidata_qid)
            WHERE wikidata_qid IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_em_canonical_country_date
            ON entity_mentions(country_code, collected_date, entity_type, canonical);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_em_canonical_country_date;
        DROP INDEX IF EXISTS idx_entity_resolution_cache_qid;
        DROP TABLE IF EXISTS entity_resolution_cache;

        ALTER TABLE entity_mentions
            DROP COLUMN IF EXISTS resolver_method,
            DROP COLUMN IF EXISTS resolution_confidence;

        ALTER TABLE news_items
            DROP COLUMN IF EXISTS calibration_method,
            DROP COLUMN IF EXISTS calibrated_sentiment_scores,
            DROP COLUMN IF EXISTS calibrated_sentiment_score,
            DROP COLUMN IF EXISTS raw_sentiment_score;
        """
    )
