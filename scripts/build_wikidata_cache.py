"""Preemptively warm the Wikidata entity resolution cache.

Resolves the highest-volume still-unresolved (entity_text, entity_type) pairs
ahead of the daily pipeline so the live ``entity_resolution`` step hits the
cache instead of the Wikidata API. Local aliases short-circuit without any
network call; the rest go through the exact same Wikidata path the resolver
uses, then everything is upserted into ``entity_resolution_cache``.

Usage:
    python scripts/build_wikidata_cache.py --country germany --since 2026-05-10 --limit 300
    python scripts/build_wikidata_cache.py --country germany --date 2026-05-14
    python scripts/build_wikidata_cache.py --country turkey --since 2026-05-01 --sleep 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from typing import Any

from loguru import logger

from src.analysis.entity_canonicalization import canonicalize_mention, normalize_entity_text
from src.analysis.entity_resolution import _resolve_wikidata
from src.config import load_country_config
from src.db.queries import (
    fetch_top_entity_texts_for_cache,
    upsert_entity_resolution_cache,
)


def build_wikidata_cache(
    country_config: dict[str, Any],
    since_date: str,
    limit: int = 500,
    sleep: float = 1.0,
) -> dict[str, int]:
    country_code = country_config["country_code"]
    language = country_config.get("language")
    cfg = country_config.get("entity_narrative") or {}
    wikidata_cfg = cfg.get("wikidata") or {}
    wikidata_enabled = bool(wikidata_cfg.get("enabled", False))
    min_confidence = float(wikidata_cfg.get("min_confidence", 0.85))

    if not wikidata_enabled:
        logger.warning(
            f"wikidata disabled for [{country_code}] — caching local-alias "
            "canonicals only, no network calls"
        )

    pairs = fetch_top_entity_texts_for_cache(
        country_code=country_code,
        since_date=since_date,
        limit=limit,
    )
    if not pairs:
        logger.info(
            f"No unresolved entity texts since {since_date} [{country_code}]"
        )
        return {"total": 0, "resolved_qid": 0, "local_alias": 0, "miss": 0}

    logger.info(
        f"Warming cache for {len(pairs)} entity texts since {since_date} "
        f"[{country_code}], wikidata_enabled={wikidata_enabled}"
    )

    cache_rows: list[dict[str, Any]] = []
    stats = {"total": len(pairs), "resolved_qid": 0, "local_alias": 0, "miss": 0}

    for idx, pair in enumerate(pairs, start=1):
        entity_text = pair["entity_text"]
        entity_type = pair["entity_type"]
        try:
            local = canonicalize_mention(entity_text, entity_type, country_config)
            if local.resolver_method == "local_alias":
                resolved = local
                stats["local_alias"] += 1
            elif wikidata_enabled:
                wikidata = _resolve_wikidata(
                    entity_text,
                    entity_type,
                    country_config,
                    min_confidence=min_confidence,
                )
                time.sleep(sleep)
                if wikidata is not None:
                    resolved = wikidata
                    stats["resolved_qid"] += 1
                else:
                    resolved = local
                    stats["miss"] += 1
            else:
                resolved = local
                stats["miss"] += 1

            cache_rows.append(
                {
                    "normalized_text": normalize_entity_text(entity_text, language),
                    "entity_type": entity_type,
                    "country_code": country_code,
                    "canonical": resolved.canonical,
                    "wikidata_qid": resolved.wikidata_qid,
                    "confidence": resolved.confidence,
                    "resolver_method": resolved.resolver_method,
                }
            )
        except Exception as exc:  # fail-soft per entity
            logger.warning(
                f"cache warm failed soft for {entity_text!r} ({entity_type}): {exc}"
            )
            continue

        if idx % 25 == 0 or idx == len(pairs):
            logger.info(f"  progress {idx}/{len(pairs)}")

    upsert_entity_resolution_cache(cache_rows)
    logger.success(
        f"wikidata cache warm done [{country_code}] — "
        f"total={stats['total']} resolved_qid={stats['resolved_qid']} "
        f"local_alias={stats['local_alias']} miss={stats['miss']}"
    )
    return stats


def _default_since() -> str:
    return (date.today() - timedelta(days=14)).isoformat()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preemptively warm the Wikidata entity resolution cache."
    )
    parser.add_argument("--country", required=True, help="Country slug or ISO code")
    parser.add_argument(
        "--since",
        default=_default_since(),
        metavar="YYYY-MM-DD",
        help="Lower bound on collected_date (default: 14 days ago)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max distinct entity texts to warm (default 500)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between Wikidata API calls (default 1.0)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Convenience: single date; overrides --since",
    )
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.sleep < 0:
        parser.error("--sleep must be >= 0")
    if args.date:
        args.since = args.date
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    country_config = load_country_config(args.country)
    build_wikidata_cache(
        country_config=country_config,
        since_date=args.since,
        limit=args.limit,
        sleep=args.sleep,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
