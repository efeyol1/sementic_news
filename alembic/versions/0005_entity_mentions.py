"""entity_mentions table

First building block of the entity-centric narrative track (Sprint 1).
Stores one row per detected entity mention so downstream collocation
analysis (Sprint 5) can rebuild context windows from positions. Lives
alongside the legacy ``news_items.entities`` JSONB column — the older
NER feature stays untouched until Sprint 9 deprecation.

Schema notes:
  * ``country_code`` and ``collected_date`` are denormalized from
    ``news_items`` so cross-country queries (Sprint 7+) avoid a join.
  * ``wikidata_qid`` / ``canonical`` are nullable — Sprint 1 fills only
    raw NER output; Sprint 2 adds Wikidata linking and back-fills.
  * ``position_in_article`` is a token offset (filled when available).
    Collocation extraction (Sprint 5) walks ±N tokens around it.
  * The partial index on ``wikidata_qid`` skips unresolved mentions so
    Sprint 2+ profile queries stay cheap.

Revision ID: 0005_entity_mentions
Revises: 0004_country_derived
Create Date: 2026-05-14
"""

from __future__ import annotations

from alembic import op

revision: str = "0005_entity_mentions"
down_revision: str | None = "0004_country_derived"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_mentions (
            id                  BIGSERIAL PRIMARY KEY,
            article_id          INTEGER NOT NULL
                                  REFERENCES news_items(id) ON DELETE CASCADE,
            country_code        VARCHAR(2) NOT NULL,
            collected_date      DATE NOT NULL,
            entity_text         TEXT NOT NULL,
            entity_type         VARCHAR(8) NOT NULL,
            wikidata_qid        VARCHAR(16),
            canonical           TEXT,
            position_in_article INTEGER,
            extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_em_country_date
            ON entity_mentions(country_code, collected_date);

        CREATE INDEX IF NOT EXISTS idx_em_article
            ON entity_mentions(article_id);

        CREATE INDEX IF NOT EXISTS idx_em_qid_country_date
            ON entity_mentions(wikidata_qid, country_code, collected_date)
            WHERE wikidata_qid IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS idx_em_qid_country_date;
        DROP INDEX IF EXISTS idx_em_article;
        DROP INDEX IF EXISTS idx_em_country_date;
        DROP TABLE IF EXISTS entity_mentions;
        """
    )
