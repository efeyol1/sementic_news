"""Full daily pipeline orchestrator.

Runs the complete chain for a given date:
    collect → article_fetch(optional) → preprocess → sentiment → ner → clustering → vector_store → drift

The trailing ``drift`` step is fail-soft instrumentation: it never blocks
the pipeline, but it writes ``data/drift_reports/<date>.json`` and (under
GitHub Actions) appends a PSI summary to ``$GITHUB_STEP_SUMMARY``.

Usage:
    python -m src.pipeline                    # run today's full pipeline
    python -m src.pipeline --country turkey
    python -m src.pipeline --date 2026-04-20
    python -m src.pipeline --skip-collect     # skip RSS fetch (raw file exists)
    python -m src.pipeline --fetch-articles --article-limit 100
    python -m src.pipeline --n-clusters 20
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

from loguru import logger

from src.analysis.clustering import cluster_topics
from src.analysis.collocation_stats import compute_collocation_stats_batch
from src.analysis.collocations import extract_collocations_batch
from src.analysis.entity_extraction import extract_entities_batch
from src.analysis.entity_resolution import resolve_entities_batch
from src.analysis.ner import extract_entities
from src.analysis.sentiment import analyze
from src.analysis.sentiment_calibration import calibrate_date
from src.analysis.vector_store import index_date
from src.config import load_country_config
from src.data.article_fetcher import fetch_articles
from src.data.preprocessor import preprocess
from src.data.rss_collector import collect_all
from src.monitoring.drift import run_drift_check

# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_COUNTRY = "turkey"


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


def _section_enabled(country_config: dict, section: str, default: bool = True) -> bool:
    payload = country_config.get(section) or {}
    return bool(payload.get("enabled", default))


def _validate_sentiment_backend(country_config: dict) -> None:
    """Prevent accidental cross-language use of a fine-tuned sentiment model."""
    sentiment_cfg = country_config.get("sentiment") or {}
    configured_finetuned = sentiment_cfg.get("finetuned_model")
    env_model = os.getenv("SENTIMENT_MODEL_ID")
    env_backend = os.getenv("SENTIMENT_BACKEND", "").lower()
    if configured_finetuned:
        return
    if env_model or env_backend == "onnx_int8":
        raise RuntimeError(
            "Sentiment fine-tuned backend is active, but this country config "
            "does not declare sentiment.finetuned_model. Unset SENTIMENT_MODEL_ID/"
            "SENTIMENT_BACKEND to use zero-shot, or configure a country-specific "
            "fine-tuned model."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    date_str: str | None = None,
    country: str = _DEFAULT_COUNTRY,
    skip_collect: bool = False,
    fetch_article_bodies: bool = False,
    article_limit: int = 100,
    article_workers: int = 8,
    n_clusters: int = 15,
) -> None:
    """Execute the full daily pipeline for *date_str*.

    Args:
        date_str: ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        country: Country slug or ISO code from ``configs/countries/*.yaml``.
        skip_collect: If True, skip RSS collection (raw file must exist).
        fetch_article_bodies: If True, fetch article body text before preprocessing.
        article_limit: Max article bodies to fetch in this run.
        article_workers: Concurrent article fetch workers.
        n_clusters: Number of topic clusters for the clustering step.
    """
    date_str = date_str or date.today().isoformat()
    country_config = load_country_config(country)
    country_code = country_config["country_code"]
    country_slug = country_config["country_slug"]
    language = country_config["language"]

    logger.info(
        f"Pipeline start — date={date_str}, country={country_slug} "
        f"[{country_code}/{language}], skip_collect={skip_collect}"
    )
    wall_start = time.perf_counter()

    if not skip_collect:
        _step("collect", collect_all, date_str=date_str, country=country_slug)

    if fetch_article_bodies:
        _step(
            "article_fetch",
            fetch_articles,
            date_str=date_str,
            limit=article_limit,
            workers=article_workers,
            ensure_schema=True,
            country_code=country_code,
            country_slug=country_slug,
        )

    _step(
        "preprocess",
        preprocess,
        date_str=date_str,
        country_code=country_code,
        target_language=language,
    )
    if _section_enabled(country_config, "sentiment"):
        sentiment_cfg = country_config.get("sentiment") or {}
        calibration_enabled = bool(
            (sentiment_cfg.get("calibration") or {}).get("enabled", False)
        )
        _validate_sentiment_backend(country_config)
        _step(
            "sentiment",
            analyze,
            date_str=date_str,
            lang=language,
            country_code=country_code,
            zero_shot_model=sentiment_cfg.get("zero_shot_model"),
            candidate_labels=sentiment_cfg.get("candidate_labels"),
            calibration_enabled=calibration_enabled,
        )
        if calibration_enabled:
            _step(
                "sentiment_calibration",
                calibrate_date,
                date_str=date_str,
                country_config=country_config,
            )
        else:
            logger.info("Skipping sentiment_calibration — disabled in country config")
    else:
        logger.info("Skipping sentiment — disabled in country config")

    if _section_enabled(country_config, "ner"):
        _step("ner", extract_entities, date_str=date_str, country_code=country_code)
    else:
        logger.info("Skipping ner — disabled in country config")

    # Entity-narrative track (Sprint 1+). Default false so legacy configs
    # without the section stay opt-out; ``germany.yaml`` opts in today.
    if _section_enabled(country_config, "entity_narrative", default=False):
        _step(
            "entity_extraction",
            extract_entities_batch,
            date_str=date_str,
            country_config=country_config,
        )
        _step_soft(
            "entity_resolution",
            resolve_entities_batch,
            date_str=date_str,
            country_config=country_config,
        )
        # Sprint 5: per-mention collocation extraction. Soft so a spaCy
        # model load failure in one country does not block clustering or
        # downstream steps for the rest of the daily run.
        collocations_cfg = (
            (country_config.get("entity_narrative") or {}).get("collocations") or {}
        )
        if collocations_cfg.get("enabled", False):
            _step_soft(
                "entity_collocations",
                extract_collocations_batch,
                date_str=date_str,
                country_config=country_config,
            )
            # Sprint 6: backfill PMI / LLR / totals over the same slice.
            # Implicit gate — runs whenever extraction is on; no separate
            # config flag (downstream consumers in Sprint 7/8 always want
            # the stats so a config to disable would just create dead data).
            _step_soft(
                "entity_pmi",
                compute_collocation_stats_batch,
                date_str=date_str,
                country_config=country_config,
            )
        else:
            logger.info(
                "Skipping entity_collocations — disabled in country config"
            )
    else:
        logger.info("Skipping entity_extraction — disabled in country config")

    if _section_enabled(country_config, "clustering"):
        _step(
            "clustering",
            cluster_topics,
            date_str=date_str,
            n_clusters=n_clusters,
            country_code=country_code,
            country_config=country_config,
        )
    else:
        logger.info("Skipping clustering — disabled in country config")

    if _section_enabled(country_config, "embeddings"):
        _step("vector_store", index_date, date_str=date_str, country_code=country_code)
    else:
        logger.info("Skipping vector_store — embeddings disabled in country config")
    _step_soft("drift", run_drift_check, target_date=date_str, country_code=country_code)

    total = time.perf_counter() - wall_start
    logger.success(
        f"Pipeline complete — {date_str} [{country_code}/{country_slug}] "
        f"finished in {total:.1f}s"
    )


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
        "--country",
        default=_DEFAULT_COUNTRY,
        help="Country slug or ISO code from configs/countries/*.yaml (default: turkey)",
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
            country=args.country,
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
