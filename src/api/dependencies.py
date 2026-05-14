"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query

from src.config.country_loader import load_country_config


async def resolve_country(
    country: str = Query(
        "turkey",
        description="Country slug (e.g. 'turkey') or ISO code (e.g. 'TR'). Default: turkey.",
    ),
) -> dict[str, Any]:
    """Resolve a country slug/code to its config dict, or 404 if unknown.

    Used as ``Depends(resolve_country)`` on country-scoped endpoints; pull
    ``country_code`` from the returned dict for DB queries.
    """
    try:
        return load_country_config(country)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Country '{country}' not found. "
                "See /api/countries for available options."
            ),
        ) from exc
