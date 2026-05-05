"""article body ingestion fields

Adds optional article-body and discovery metadata fields to ``news_items``.
The existing RSS/title-summary pipeline remains valid when these fields are
NULL; the article fetcher can backfill them incrementally.

Revision ID: 0002_article_body_ingestion
Revises: 0001_baseline
Create Date: 2026-05-05
"""

from __future__ import annotations

from alembic import op

revision: str = "0002_article_body_ingestion"
down_revision: str | None = "0001_baseline"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE news_items
            ADD COLUMN IF NOT EXISTS article_text TEXT,
            ADD COLUMN IF NOT EXISTS cleaned_article_text TEXT,
            ADD COLUMN IF NOT EXISTS canonical_category TEXT,
            ADD COLUMN IF NOT EXISTS discovery_role TEXT,
            ADD COLUMN IF NOT EXISTS parse_status TEXT,
            ADD COLUMN IF NOT EXISTS parse_error TEXT,
            ADD COLUMN IF NOT EXISTS article_fetched_at TIMESTAMPTZ;

        CREATE INDEX IF NOT EXISTS idx_news_parse_status
            ON news_items(parse_status, collected_date);

        CREATE INDEX IF NOT EXISTS idx_news_canonical_category
            ON news_items(canonical_category, collected_date);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_news_canonical_category;
        DROP INDEX IF EXISTS idx_news_parse_status;

        ALTER TABLE news_items
            DROP COLUMN IF EXISTS article_fetched_at,
            DROP COLUMN IF EXISTS parse_error,
            DROP COLUMN IF EXISTS parse_status,
            DROP COLUMN IF EXISTS discovery_role,
            DROP COLUMN IF EXISTS canonical_category,
            DROP COLUMN IF EXISTS cleaned_article_text,
            DROP COLUMN IF EXISTS article_text;
        """
    )
