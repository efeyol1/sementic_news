"""Smoke tests for the Alembic migration setup.

These tests exercise *config* and *script discovery*, not actual SQL
execution: running the baseline migration requires a real Postgres with
the ``vector`` extension, which is too heavy for a unit-test job. The
weekly retrain workflow runs the migration end-to-end against the live
DB; these tests catch the cheap mistakes (missing revision, broken
script_location, mis-imported env.py) before they reach prod.
"""

from __future__ import annotations

import os

import pytest
from alembic.script import ScriptDirectory

# DATABASE_URL is required by alembic/env.py at import time. The tests
# never actually connect — a syntactically valid placeholder URL is
# enough for ScriptDirectory to walk the migrations directory.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg2://test:test@localhost:5432/test"
)

from src.db.schema import _alembic_config  # noqa: E402


@pytest.fixture
def script_dir() -> ScriptDirectory:
    return ScriptDirectory.from_config(_alembic_config())


def test_alembic_has_exactly_one_head(script_dir: ScriptDirectory):
    """Multiple heads = a merge migration is missing. Catch it early."""
    heads = script_dir.get_heads()
    assert len(heads) == 1, f"expected single head, got {heads}"


def test_baseline_revision_present(script_dir: ScriptDirectory):
    """The baseline must remain reachable; renaming it would orphan prod
    DBs that already have ``alembic_version = '0001_baseline'``."""
    revisions = {r.revision for r in script_dir.walk_revisions()}
    assert "0001_baseline" in revisions


def test_country_derived_revision_present(script_dir: ScriptDirectory):
    """Prod DBs may already be stamped with this revision; keep it reachable."""
    revisions = {r.revision for r in script_dir.walk_revisions()}
    assert "0004_country_derived" in revisions


def test_baseline_is_root(script_dir: ScriptDirectory):
    """Baseline must have no down_revision (it's the schema starting
    point). Anything else means we accidentally chained off the wrong
    revision."""
    base = script_dir.get_revision("0001_baseline")
    assert base.down_revision is None


def test_script_location_resolves():
    """``_alembic_config`` should set ``script_location`` to an absolute
    path that exists, regardless of the cwd the caller invoked from."""
    cfg = _alembic_config()
    location = cfg.get_main_option("script_location")
    assert location and os.path.isdir(location), location
    assert os.path.isfile(os.path.join(location, "env.py"))
