"""Pytest setup shared across the test suite.

Two responsibilities:

1. **scripts/ on sys.path.** ``scripts/`` lives outside the installed package,
   so it isn't importable by default. Splice it onto ``sys.path`` once for
   every test session instead of repeating the dance in each test module.

2. **Behavioral → MLflow bridge.** ``pytest_runtest_logreport`` aggregates
   outcomes for every test marked ``@pytest.mark.behavioral`` (passed /
   failed / xfailed / xpassed), bucketed by category prefix
   (``must_pass`` / ``watchlist`` / ``aspirational``). On session end the
   counts are flushed to MLflow as the ``news-sentiment-behavioral``
   experiment — but only when ``MLFLOW_TRACKING_URI`` is set. This means
   ``weekly_retrain.yml`` (which has the secret) records bias metrics
   against each promoted model, while local / PR runs stay quiet.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# (category, outcome) -> count. Module-level so the hooks can mutate it.
_behavioral_results: dict[tuple[str, str], int] = defaultdict(int)


def _category(test_name: str) -> str:
    if test_name.startswith("test_must_pass"):
        return "must_pass"
    if test_name.startswith("test_watchlist"):
        return "watchlist"
    if test_name.startswith("test_should_pass"):
        return "aspirational"
    return "other"


def pytest_runtest_logreport(report) -> None:
    """Aggregate behavioral outcomes during the call phase only.

    Pytest emits a logreport for setup/call/teardown phases — we only care
    about ``call`` so each test counts once. ``wasxfail`` is set by pytest
    whenever an ``@pytest.mark.xfail`` test runs; combined with the actual
    pass/fail outcome it disambiguates xfailed (expected fail) from xpassed
    (unexpectedly passed — a watchlist promotion signal).
    """
    if "behavioral" not in report.keywords:
        return
    if report.when != "call":
        return

    if hasattr(report, "wasxfail"):
        outcome = "xpassed" if report.outcome == "passed" else "xfailed"
    else:
        outcome = report.outcome  # "passed" / "failed" / "skipped"

    # nodeid format: "tests/behavioral/test_X.py::test_func[param-id]"
    test_name = report.nodeid.split("::")[1].split("[")[0]
    _behavioral_results[(_category(test_name), outcome)] += 1


def pytest_sessionfinish(session, exitstatus) -> None:
    """Flush behavioral metrics to MLflow at session end.

    Silent no-op when no behavioral tests ran (unit-only sessions) or
    when ``MLFLOW_TRACKING_URI`` is unset (local dev / CI without the
    secret) — the bridge is opt-in via env presence.
    """
    if not _behavioral_results:
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow
    except ImportError:
        # mlflow lives in the ``pipeline`` extra; weekly_retrain has it,
        # but be defensive in case the hook fires somewhere it shouldn't.
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("news-sentiment-behavioral")

    # Per-(category, outcome) counts plus a per-category total so dashboards
    # can compute pass-rate without re-summing.
    metrics: dict[str, int] = {}
    totals: dict[str, int] = defaultdict(int)
    for (cat, outcome), count in _behavioral_results.items():
        metrics[f"{cat}_{outcome}"] = count
        totals[f"{cat}_total"] += count
    metrics.update(totals)

    sha = (os.environ.get("GITHUB_SHA") or "local")[:7]
    run_name = f"behavioral-{sha}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("type", "behavioral")
        mlflow.set_tag(
            "model_id",
            os.environ.get("BEHAVIORAL_MODEL_ID", "efeyol11/bert-turkish-sentiment"),
        )
        if os.environ.get("GITHUB_RUN_ID"):
            mlflow.set_tag("github_run_id", os.environ["GITHUB_RUN_ID"])
        if os.environ.get("GITHUB_WORKFLOW"):
            mlflow.set_tag("github_workflow", os.environ["GITHUB_WORKFLOW"])
        mlflow.log_metrics(metrics)
