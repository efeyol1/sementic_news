"""Backfill: run the full pipeline for a range of past dates.

Usage:
    # Last 7 days
    python -m src.backfill --days 7

    # Specific date range
    python -m src.backfill --start 2026-04-01 --end 2026-04-24

    # Dry-run (show which dates would be processed, skip already-done ones)
    python -m src.backfill --days 30 --dry-run

    # Re-process even if data exists
    python -m src.backfill --days 7 --force
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

from loguru import logger

from src.db.queries import fetch_available_dates
from src.pipeline import run


def _date_range(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def backfill(
    start: date,
    end: date,
    force: bool = False,
    dry_run: bool = False,
    n_clusters: int = 15,
    skip_collect: bool = False,
) -> None:
    dates = _date_range(start, end)
    already_done = set(fetch_available_dates()) if not force else set()

    pending = [d for d in dates if d.isoformat() not in already_done]

    if not pending:
        logger.info("Nothing to backfill — all dates already in DB. Use --force to reprocess.")
        return

    logger.info(f"Backfill: {len(pending)} dates to process ({start} → {end})")
    if dry_run:
        for d in pending:
            print(d.isoformat())
        return

    wall_start = time.perf_counter()
    success, failed = 0, []

    for d in pending:
        date_str = d.isoformat()
        logger.info(f"═══ Backfilling {date_str} ({'skip' if d.isoformat() in already_done else 'new'}) ═══")
        try:
            run(date_str=date_str, n_clusters=n_clusters, skip_collect=skip_collect)
            success += 1
        except Exception as exc:
            logger.error(f"Failed {date_str}: {exc}")
            failed.append(date_str)

    elapsed = time.perf_counter() - wall_start
    logger.success(
        f"Backfill complete: {success} OK, {len(failed)} failed in {elapsed:.1f}s"
    )
    if failed:
        logger.warning(f"Failed dates: {failed}")
        sys.exit(1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill pipeline for a date range")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--days", type=int, metavar="N", help="Last N days (including today)")
    group.add_argument("--start", metavar="YYYY-MM-DD", help="Start date (use with --end)")
    p.add_argument("--end", metavar="YYYY-MM-DD", default=date.today().isoformat())
    p.add_argument("--force", action="store_true", help="Reprocess even if date exists in DB")
    p.add_argument("--dry-run", action="store_true", help="Print dates to process without running")
    p.add_argument("--n-clusters", type=int, default=15)
    p.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip RSS collection (re-run analysis on existing rows)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    end_date = date.fromisoformat(args.end)
    if args.days:
        start_date = end_date - timedelta(days=args.days - 1)
    else:
        start_date = date.fromisoformat(args.start)

    if start_date > end_date:
        logger.error("--start must be before --end")
        sys.exit(1)

    backfill(
        start=start_date,
        end=end_date,
        force=args.force,
        dry_run=args.dry_run,
        n_clusters=args.n_clusters,
        skip_collect=args.skip_collect,
    )
