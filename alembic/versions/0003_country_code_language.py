"""country code and language columns

Adds the multi-country foundation: ``country_code`` and ``language`` on
``news_items``. Existing rows are backfilled with the V1 pilot values
(``TR`` / ``tr``) via the column DEFAULT — PostgreSQL 11+ stores the
default in metadata and applies it without rewriting the table, so the
migration is atomic and safe to run while the daily pipeline is idle.

Revision ID: 0003_country_code_language
Revises: 0002_article_body_ingestion
Create Date: 2026-05-08
"""

from __future__ import annotations

from alembic import op

revision: str = "0003_country_code_language"
down_revision: str | None = "0002_article_body_ingestion"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE news_items
            ADD COLUMN IF NOT EXISTS country_code VARCHAR(2)  NOT NULL DEFAULT 'TR',
            ADD COLUMN IF NOT EXISTS language     VARCHAR(8)  NOT NULL DEFAULT 'tr';

        CREATE INDEX IF NOT EXISTS idx_news_country_date
            ON news_items(country_code, collected_date);

        CREATE INDEX IF NOT EXISTS idx_news_language
            ON news_items(language);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_news_language;
        DROP INDEX IF EXISTS idx_news_country_date;

        ALTER TABLE news_items
            DROP COLUMN IF EXISTS language,
            DROP COLUMN IF EXISTS country_code;
        """
    )
