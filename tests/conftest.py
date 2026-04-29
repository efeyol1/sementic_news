"""Pytest setup shared across the test suite.

``scripts/`` lives outside the installed package, so it isn't importable
by default. Splice it onto ``sys.path`` here once for every test session
instead of repeating the dance in each test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
