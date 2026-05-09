"""country-scoped derived analysis tables

Phase 4 added country metadata to raw ``news_items``. This follow-up makes
derived daily artifacts safe before Phase 3 wires ``--country`` through the
pipeline: existing callers still default to the Turkey pilot, while future
country runs can write cluster and drift outputs without overwriting each other.

Revision ID: 0004_country_derived
Revises: 0003_country_code_language
Create Date: 2026-05-08
"""

from __future__ import annotations

from alembic import op

revision: str = "0004_country_derived"
down_revision: str | None = "0003_country_code_language"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE cluster_summaries
            ADD COLUMN IF NOT EXISTS country_code VARCHAR(2) NOT NULL DEFAULT 'TR';

        ALTER TABLE cluster_summaries
            DROP CONSTRAINT IF EXISTS cluster_summaries_date_cluster_id_key;

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'cluster_summaries_country_date_cluster_key'
            ) THEN
                ALTER TABLE cluster_summaries
                    ADD CONSTRAINT cluster_summaries_country_date_cluster_key
                    UNIQUE(country_code, date, cluster_id);
            END IF;
        END $$;

        CREATE INDEX IF NOT EXISTS idx_cluster_country_date
            ON cluster_summaries(country_code, date);

        ALTER TABLE drift_reports
            ADD COLUMN IF NOT EXISTS country_code VARCHAR(2) NOT NULL DEFAULT 'TR';

        ALTER TABLE drift_reports
            DROP CONSTRAINT IF EXISTS drift_reports_pkey;

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'drift_reports_country_date_pkey'
            ) THEN
                ALTER TABLE drift_reports
                    ADD CONSTRAINT drift_reports_country_date_pkey
                    PRIMARY KEY(country_code, date);
            END IF;
        END $$;

        CREATE INDEX IF NOT EXISTS idx_drift_country_date
            ON drift_reports(country_code, date DESC);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_drift_country_date;
        DROP INDEX IF EXISTS idx_cluster_country_date;

        ALTER TABLE drift_reports
            DROP CONSTRAINT IF EXISTS drift_reports_country_date_pkey;

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'drift_reports_pkey'
            ) THEN
                ALTER TABLE drift_reports
                    ADD CONSTRAINT drift_reports_pkey PRIMARY KEY(date);
            END IF;
        END $$;

        ALTER TABLE cluster_summaries
            DROP CONSTRAINT IF EXISTS cluster_summaries_country_date_cluster_key;

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'cluster_summaries_date_cluster_id_key'
            ) THEN
                ALTER TABLE cluster_summaries
                    ADD CONSTRAINT cluster_summaries_date_cluster_id_key
                    UNIQUE(date, cluster_id);
            END IF;
        END $$;

        ALTER TABLE drift_reports
            DROP COLUMN IF EXISTS country_code;

        ALTER TABLE cluster_summaries
            DROP COLUMN IF EXISTS country_code;
        """
    )
