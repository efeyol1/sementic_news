"""entity_country_profile table

Sprint 7 of the entity-narrative track: rolling 7-day and 30-day
profile rows per (entity, country). One row per
``(country_code, canonical, entity_type, window_days, end_date)``;
the daily aggregator (`src/analysis/country_profile.py`) writes two
rows per entity each day (window_days = 7 and 30).

Sprint 6 left ``entity_collocations`` with PMI/LLR filled per day.
Sprint 7 sums daily cells over the window and *recomputes* PMI/LLR
from window-level marginals (aggregate-first, see plan §4). Top-20
collocates are persisted as JSONB ranked by LLR desc → PMI desc →
lemma asc. Partial windows (e.g. TR which has just been activated)
emit a row with ``coverage_days < window_days``; downstream
consumers can filter or render "partial" badges.

Schema notes:
  * UNIQUE(country_code, canonical, entity_type, window_days, end_date)
    is the natural per-day profile key. The Sprint 7 batch helper
    deletes by (country_code, end_date) and re-inserts so a re-run is
    idempotent at the daily granularity.
  * ``wikidata_qid`` is nullable for the same reason as Sprint 5's
    ``entity_collocations``: unresolved mentions still get a row keyed
    by their canonical text. The partial index speeds up Sprint 8's
    ``/api/entity/{qid}/profile`` resolved-only lookup.
  * ``cooccurrence_with_entity_total`` mirrors Sprint 6's column name
    for symmetry — it carries the same window-summed R(e) value as
    ``total_cooccurrences``; we keep both so Sprint 8 serializers can
    use whichever name reads better in their context.

Revision ID: 0008_entity_country_profile
Revises: 0007_entity_collocations
Create Date: 2026-05-29
"""

from __future__ import annotations

from alembic import op

revision: str = "0008_entity_country_profile"
down_revision: str | None = "0007_entity_collocations"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_country_profile (
            id                              BIGSERIAL PRIMARY KEY,
            country_code                    VARCHAR(2) NOT NULL,
            canonical                       TEXT NOT NULL,
            entity_type                     VARCHAR(8) NOT NULL,
            wikidata_qid                    VARCHAR(16),
            window_days                     SMALLINT NOT NULL,
            end_date                        DATE NOT NULL,
            coverage_days                   SMALLINT NOT NULL,

            total_cooccurrences             BIGINT NOT NULL,
            total_mentions                  BIGINT NOT NULL,
            cooccurrence_with_entity_total  BIGINT NOT NULL,
            window_total                    BIGINT NOT NULL,

            avg_pmi                         DOUBLE PRECISION,
            avg_log_likelihood              DOUBLE PRECISION,

            top_collocates                  JSONB NOT NULL,

            profile_computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            UNIQUE (country_code, canonical, entity_type,
                    window_days, end_date)
        );

        CREATE INDEX IF NOT EXISTS idx_ecp_qid_window
            ON entity_country_profile(wikidata_qid, window_days, end_date)
            WHERE wikidata_qid IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_ecp_country_window_end
            ON entity_country_profile(country_code, window_days, end_date);

        CREATE INDEX IF NOT EXISTS idx_ecp_country_end_canonical
            ON entity_country_profile(country_code, end_date, canonical);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_ecp_country_end_canonical;
        DROP INDEX IF EXISTS idx_ecp_country_window_end;
        DROP INDEX IF EXISTS idx_ecp_qid_window;
        DROP TABLE IF EXISTS entity_country_profile;
        """
    )
