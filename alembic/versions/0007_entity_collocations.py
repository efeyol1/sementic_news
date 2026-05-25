"""entity_collocations table

Sprint 5 of the entity-narrative track: per-day, per-country, per-entity
lemma cooccurrence counts. The raw ``cooccurrence_count`` is the only
column Sprint 5 fills; Sprint 6 backfills ``pmi`` / ``log_likelihood``
plus the four ``*_total`` columns it needs to compute the statistics
in-place via ``UPDATE``. Creating both sets of columns in a single
migration keeps the rollback story trivial (one DROP TABLE).

Schema notes:
  * UNIQUE(country_code, collected_date, canonical, entity_type, lemma, pos)
    is the natural daily aggregation key. The ``bulk_upsert`` helper
    deletes by (country_code, collected_date) and re-inserts so a
    re-run produces identical rows.
  * ``wikidata_qid`` is nullable because some mentions never resolve;
    we keep them under their ``canonical`` text so the row is still
    addressable from the API. The partial index speeds up the
    resolved-only path used by /api/entity/{qid}/profile (Sprint 8).
  * ``canonical`` is denormalized from ``entity_mentions`` so the
    Sprint 6 PMI job (and Sprint 7 rolling aggregator) can read the
    daily counts without re-joining mentions.

Revision ID: 0007_entity_collocations
Revises: 0006_resolver_calibration
Create Date: 2026-05-25
"""

from __future__ import annotations

from alembic import op

revision: str = "0007_entity_collocations"
down_revision: str | None = "0006_resolver_calibration"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_collocations (
            id                              BIGSERIAL PRIMARY KEY,
            country_code                    VARCHAR(2) NOT NULL,
            collected_date                  DATE NOT NULL,
            wikidata_qid                    VARCHAR(16),
            canonical                       TEXT NOT NULL,
            entity_type                     VARCHAR(8) NOT NULL,
            lemma                           TEXT NOT NULL,
            pos                             VARCHAR(8) NOT NULL,
            cooccurrence_count              INTEGER NOT NULL,
            cooccurrence_with_entity_total  INTEGER,
            entity_total                    INTEGER,
            token_total                     INTEGER,
            window_total                    INTEGER,
            pmi                             DOUBLE PRECISION,
            log_likelihood                  DOUBLE PRECISION,
            extracted_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (country_code, collected_date, canonical,
                    entity_type, lemma, pos)
        );

        CREATE INDEX IF NOT EXISTS idx_ec_qid_country_date
            ON entity_collocations(wikidata_qid, country_code, collected_date)
            WHERE wikidata_qid IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_ec_country_date_canonical
            ON entity_collocations(country_code, collected_date, canonical);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_ec_country_date_canonical;
        DROP INDEX IF EXISTS idx_ec_qid_country_date;
        DROP TABLE IF EXISTS entity_collocations;
        """
    )
