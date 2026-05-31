"""entity_explanation cache table (Sprint 11 — RAG explainability)

Sprint 11 adds a generation layer that explains an entity's pre-computed
framing signals (top collocates + frame_intensities from
``entity_country_profile``) in grounded natural language. Generating that
text — whether via the Claude API or the deterministic extractive
fallback — is cached here so we generate **once per profile version** and
never pay (API cost or latency) for an unchanged profile.

Cache-key semantics:
  * UNIQUE(country_code, canonical, entity_type, window_days, end_date, lang)
    is the *identity* of an explanation (one per entity/country/window/
    end_date/language).
  * ``profile_hash`` is the *validity check*, NOT part of the key. A read
    by the unique key is a hit only when the stored ``profile_hash`` still
    equals the hash of the current profile row. When the Sprint 7
    aggregator rewrites the profile, the hash changes → the next request
    regenerates and upserts (ON CONFLICT DO UPDATE).
  * ``backend`` records which generator produced the row
    (``anthropic`` | ``extractive``) so the dashboard can badge it.

Revision ID: 0010_entity_explanation
Revises: 0009_frame_intensities
Create Date: 2026-05-29
"""

from __future__ import annotations

from alembic import op

revision: str = "0010_entity_explanation"
down_revision: str | None = "0009_frame_intensities"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_explanation (
            id              BIGSERIAL PRIMARY KEY,
            country_code    VARCHAR(2)  NOT NULL,
            wikidata_qid    VARCHAR(16),
            canonical       TEXT        NOT NULL,
            entity_type     VARCHAR(8)  NOT NULL,
            window_days     SMALLINT    NOT NULL,
            end_date        DATE        NOT NULL,
            profile_hash    CHAR(64)    NOT NULL,
            backend         VARCHAR(16) NOT NULL,
            model           VARCHAR(48),
            lang            VARCHAR(8)  NOT NULL,
            explanation     JSONB       NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            UNIQUE (country_code, canonical, entity_type,
                    window_days, end_date, lang)
        );

        CREATE INDEX IF NOT EXISTS idx_eexp_qid_lookup
            ON entity_explanation(wikidata_qid, window_days, end_date, lang)
            WHERE wikidata_qid IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_eexp_qid_lookup;
        DROP TABLE IF EXISTS entity_explanation;
        """
    )
