"""Alembic runtime configuration.

Reads ``DATABASE_URL`` from the environment (loaded from ``.env`` for
local dev, injected as a CI/Render secret in production). The project
uses raw SQL DDL via ``op.execute(...)`` rather than SQLAlchemy ORM
models, so ``target_metadata`` stays ``None``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

load_dotenv()

config = context.config

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError(
        "DATABASE_URL is not set. Add it to .env locally or to your "
        "CI / Render environment for migrations to run."
    )
config.set_main_option("sqlalchemy.url", database_url)

# No ORM in this project — migrations are hand-written raw-SQL.
target_metadata = None


def run_migrations_offline() -> None:
    """Render migrations as SQL without connecting (for review / debugging)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against the live DB."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
