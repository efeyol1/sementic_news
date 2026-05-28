"""Country configuration loader.

V2's central principle: country-specific behavior lives in YAML, not in
Python. This module reads ``configs/countries/<slug>.yaml`` and returns
a validated dict. Adding a new country = drop a new YAML, no Python
changes required.

Public API:
    load_country_config(country: str) -> dict
    list_available_countries() -> list[dict]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.canonical_categories import (
    validate_category,
    validate_discovery_role,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs" / "countries"

# Identity fields every country config must declare. Section payloads
# (sentiment, ner, clustering, embeddings, sources) are validated lazily
# by the modules that consume them — keeping the loader minimal lets us
# evolve sub-schemas across phases without churning the loader.
_REQUIRED_FIELDS = ("country_code", "country_name", "country_slug", "language")


def _config_path(country: str) -> Path:
    """Resolve a country slug / code / file stem to an absolute YAML path.

    Accepts ``"turkey"``, ``"Turkey"``, ``"TR"``, etc. Lookup is
    case-insensitive against both the file stem (``turkey.yaml``) and
    the ``country_code`` declared inside each YAML.
    """
    needle = country.strip().lower()
    if not needle:
        raise ValueError("country must be a non-empty string")

    direct = _CONFIGS_DIR / f"{needle}.yaml"
    if direct.exists():
        return direct

    # Allow lookup by country_code (e.g. "TR" → turkey.yaml). Walk the
    # directory once; with O(10) countries this is negligible.
    for path in sorted(_CONFIGS_DIR.glob("*.yaml")):
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if cfg.get("country_code", "").lower() == needle:
            return path
        if cfg.get("country_slug", "").lower() == needle:
            return path

    available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml")) if _CONFIGS_DIR.exists() else []
    raise FileNotFoundError(
        f"Country config not found for {country!r}. "
        f"Available countries: {available or '(none — create configs/countries/*.yaml)'}."
    )


def _normalize_sources(sources: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    """Coerce each source's URL representation into a canonical ``urls`` list.

    Accepts either ``url: "single"`` (brief example form) or
    ``urls: [list]`` (multi-feed form used by NTV / Milliyet). Output is
    always a list — downstream collectors don't have to branch.
    """
    normalized: list[dict[str, Any]] = []
    for idx, src in enumerate(sources):
        if not isinstance(src, dict):
            raise ValueError(
                f"{filename}: source #{idx} must be a mapping, got {type(src).__name__}"
            )
        if "name" not in src:
            raise ValueError(f"{filename}: source #{idx} missing 'name'")

        urls = src.get("urls")
        single = src.get("url")
        if urls is None and single is None:
            raise ValueError(
                f"{filename}: source {src['name']!r} has no 'url' or 'urls'"
            )
        if urls is None:
            urls = [single]
        if not isinstance(urls, list) or not urls:
            raise ValueError(
                f"{filename}: source {src['name']!r} 'urls' must be a non-empty list"
            )

        canonical = dict(src)
        canonical["urls"] = list(urls)
        canonical.pop("url", None)
        canonical.setdefault("type", "rss")
        canonical.setdefault("enabled", True)
        canonical.setdefault("category", "general")
        canonical.setdefault("category_strategy", "infer_from_url")
        canonical.setdefault("discovery_role", "auto")
        canonical.setdefault("min_expected_items", 10)
        # Data licensing status per source. Defaults to "unknown" — most
        # publishers have not been reviewed individually. Override per
        # source with one of: "unknown", "robots-allowed", "restricted",
        # "feed-public". The authoritative inventory lives in
        # DATA_PROVENANCE.md; this field exists so future automation can
        # round-trip per-source license between YAML and the matrix.
        canonical.setdefault("license", "unknown")

        if canonical.get("canonical_category") is not None:
            validate_category(canonical["canonical_category"])
        if canonical.get("discovery_role") != "auto":
            validate_discovery_role(canonical["discovery_role"])
        if not isinstance(canonical["min_expected_items"], int) or canonical["min_expected_items"] < 0:
            raise ValueError(
                f"{filename}: source {src['name']!r} 'min_expected_items' "
                "must be a non-negative integer"
            )
        normalized.append(canonical)
    return normalized


def load_country_config(country: str) -> dict[str, Any]:
    """Load and validate a country configuration.

    Args:
        country: country slug ("turkey"), country code ("TR"), or
                 file stem. Case-insensitive.

    Returns:
        Parsed config dict with ``sources`` normalized to ``urls`` lists.

    Raises:
        FileNotFoundError: no matching YAML in ``configs/countries/``.
        ValueError: required identity field missing or source malformed.
    """
    path = _config_path(country)
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if not isinstance(config, dict):
        raise ValueError(f"{path.name}: top level must be a mapping")

    missing = [f for f in _REQUIRED_FIELDS if not config.get(f)]
    if missing:
        raise ValueError(
            f"{path.name}: missing required identity fields: {missing}"
        )

    sources = config.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"{path.name}: 'sources' must be a list, got {type(sources).__name__}")
    config["sources"] = _normalize_sources(sources, path.name)

    return config


def list_available_countries() -> list[dict[str, Any]]:
    """Enumerate every YAML in ``configs/countries/`` as an identity dict.

    Used by ``GET /api/countries`` (Phase 5). Skips broken configs so a
    single bad file doesn't break the listing endpoint.
    """
    if not _CONFIGS_DIR.exists():
        return []
    countries: list[dict[str, Any]] = []
    for path in sorted(_CONFIGS_DIR.glob("*.yaml")):
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if not all(cfg.get(f) for f in _REQUIRED_FIELDS):
            continue
        countries.append(
            {
                "code": cfg["country_code"],
                "slug": cfg["country_slug"],
                "name": cfg["country_name"],
                "language": cfg["language"],
                "status": "active",
            }
        )
    return countries
