"""frame_intensities column on entity_country_profile

Sprint 10 of the entity-narrative track: the frame bridge derives a
six-frame "framing intensity" distribution from each profile's top
collocates (see ``src/analysis/frame_bridge.py``). We persist it as one
nullable JSONB column on ``entity_country_profile`` rather than a new
table — it is a pure function of the row's ``top_collocates`` and is
written in the same Sprint 7 rollup batch.

Shape: ``{"economic": 0.42, "security": 0.31, ...}`` (L1-normalized over
the six frames) or NULL when the country has the frame bridge disabled,
or when no top collocate matched any frame seed (we never fabricate a
distribution).

Revision ID: 0009_frame_intensities
Revises: 0008_entity_country_profile
Create Date: 2026-05-29
"""

from __future__ import annotations

from alembic import op

revision: str = "0009_frame_intensities"
down_revision: str | None = "0008_entity_country_profile"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE entity_country_profile
            ADD COLUMN IF NOT EXISTS frame_intensities JSONB;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE entity_country_profile
            DROP COLUMN IF EXISTS frame_intensities;
        """
    )
