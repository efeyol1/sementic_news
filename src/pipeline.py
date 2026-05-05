"""Full daily pipeline orchestrator.

Runs the complete chain for a given date:
    collect → article_fetch(optional) → preprocess → sentiment → ner → clustering → vector_store → drift

The trailing ``drift`` step is fail-soft instrumentation: it never blocks
the pipeline, but it writes ``data/drift_reports/<date>.json`` and (under
GitHub Actions) appends a PSI summary to ``$GITHUB_STEP_SUMMARY``.

Usage:
    python -m src.pipeline                    # run today's full pipeline
    python -m src.pipeline --date 2026-04-20
    python -m src.pipeline --skip-collect     # skip RSS fetch (raw file exists)
    python -m src.pipeline --fetch-articles --article-limit 100
    python -m src.pipeline --n-clusters 20
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from loguru import logger

from src.analysis.clustering import cluster_topics
from src.analysis.ner import extract_entities
from src.analysis.sentiment import analyze
from src.analysis.vector_store import index_date
from src.data.article_fetcher import fetch_articles
from src.data.preprocessor import preprocess
from src.data.rss_collector import collect_all
from src.monitoring.drift import run_drift_check

# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _step(name: str, fn, *args, **kwargs):
    """Run a pipeline step, log timing, and propagate exceptions."""
    logger.info(f"── Step: {name} ──────────────────────────")
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.success(f"✓ {name} completed in {elapsed:.1f}s → {result}")
    return result


def _step_soft(name: str, fn, *args, **kwargs):
    """Like ``_step`` but swallows exceptions — for instrumentation that
    must never block the pipeline (e.g. drift check)."""
    logger.info(f"── Step: {name} (soft) ───────────────────")
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.warning(f"✗ {name} failed soft after {elapsed:.1f}s: {exc}")
        return None
    elapsed = time.perf_counter() - t0
    logger.success(f"✓ {name} completed in {elapsed:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    date_str: str | None = None,
    skip_collect: bool = False,
    fetch_article_bodies: bool = False,
    article_limit: int = 100,
    article_workers: int = 8,
    n_clusters: int = 15,
) -> None:
    """Execute the full daily pipeline for *date_str*.

    Args:
        date_str: ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        skip_collect: If True, skip RSS collection (raw file must exist).
        fetch_article_bodies: If True, fetch article body text before preprocessing.
        article_limit: Max article bodies to fetch in this run.
        article_workers: Concurrent article fetch workers.
        n_clusters: Number of topic clusters for the clustering step.
    """
    date_str = date_str or date.today().isoformat()

    logger.info(f"Pipeline start — date={date_str}, skip_collect={skip_collect}")
    wall_start = time.perf_counter()

    if not skip_collect:
        _step("collect", collect_all, date_str=date_str)

    if fetch_article_bodies:
        _step(
            "article_fetch",
            fetch_articles,
            date_str=date_str,
            limit=article_limit,
            workers=article_workers,
            ensure_schema=True,
        )

    _step("preprocess", preprocess, date_str=date_str)
    _step("sentiment", analyze, date_str=date_str)
    _step("ner", extract_entities, date_str=date_str)
    _step("clustering", cluster_topics, date_str=date_str, n_clusters=n_clusters)
    _step("vector_store", index_date, date_str=date_str)
    _step_soft("drift", run_drift_check, target_date=date_str)

    total = time.perf_counter() - wall_start
    logger.success(f"Pipeline complete — {date_str} finished in {total:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Semantic News daily pipeline."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date to process (default: today)",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip RSS collection step (raw file must already exist)",
    )
    parser.add_argument(
        "--fetch-articles",
        action="store_true",
        help="Fetch article body text before preprocessing (default: off)",
    )
    parser.add_argument(
        "--article-limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum article bodies to fetch when --fetch-articles is set (default: 100)",
    )
    parser.add_argument(
        "--article-workers",
        type=int,
        default=8,
        metavar="N",
        help="Concurrent article fetch workers when --fetch-articles is set (default: 8)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=15,
        metavar="N",
        help="Number of topic clusters (default: 15)",
    )
    args = parser.parse_args(argv)
    if args.article_limit < 1 or args.article_workers < 1:
        parser.error("--article-limit and --article-workers must be positive integers")
    return args


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(
            date_str=args.date,
            skip_collect=args.skip_collect,
            fetch_article_bodies=args.fetch_articles,
            article_limit=args.article_limit,
            article_workers=args.article_workers,
            n_clusters=args.n_clusters,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
