"""Database schema entry point — thin wrapper over Alembic.

Single source of truth for schema state. Every caller (the API's startup
hook, ``daily_pipeline.yml``, the CLI) goes through ``init_db()`` which
runs ``alembic upgrade head``. Migrations live in ``alembic/versions/``.

Run directly:
    python -m src.db.schema
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from loguru import logger

from alembic import command

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"


def _alembic_config() -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    # Tell Alembic where the migrations live regardless of the cwd the
    # caller invoked us from (API startup vs CI vs ``python -m``).
    cfg.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    return cfg


def init_db() -> None:
    """Apply all pending Alembic migrations. Idempotent: a no-op when the
    DB is already at head, runs the missing revisions otherwise."""
    logger.info("Running alembic upgrade head...")
    command.upgrade(_alembic_config(), "head")
    logger.success("Database schema is at head")


if __name__ == "__main__":
    init_db()
