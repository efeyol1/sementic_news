"""Canonical news category helpers for discovery.

The collector keeps source-specific feed names out of downstream analysis by
mapping each discovery URL onto this fixed category vocabulary. General feeds
such as homepage / breaking-news are marked as ``general_discovery`` rather
than treated as a topical category.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "politics_governance",
    "economy_finance",
    "world_geopolitics",
    "security_justice",
    "disaster_environment",
    "health_education_social",
    "science_technology",
    "culture_life",
    "sports",
    "magazine_entertainment",
    "other",
)

GENERAL_DISCOVERY = "general_discovery"
CATEGORY_DISCOVERY = "category"

DISCOVERY_ROLES: tuple[str, ...] = (GENERAL_DISCOVERY, CATEGORY_DISCOVERY)

_GENERAL_TOKENS = {
    "",
    "news",
    "anasayfa",
    "manset",
    "son-dakika",
    "sondakika",
    "sitemap",
    "sitemap_google_news",
}

_CATEGORY_TOKEN_MAP = {
    "gundem": "politics_governance",
    "politika": "politics_governance",
    "siyaset": "politics_governance",
    "turkiye": "politics_governance",
    "ekonomi": "economy_finance",
    "finans": "economy_finance",
    "para": "economy_finance",
    "dunya": "world_geopolitics",
    "world": "world_geopolitics",
    "dis": "world_geopolitics",
    "guvenlik": "security_justice",
    "adalet": "security_justice",
    "hukuk": "security_justice",
    "asayis": "security_justice",
    "deprem": "disaster_environment",
    "cevre": "disaster_environment",
    "afet": "disaster_environment",
    "saglik": "health_education_social",
    "egitim": "health_education_social",
    "teknoloji": "science_technology",
    "bilim": "science_technology",
    "kultur": "culture_life",
    "sanat": "culture_life",
    "yasam": "culture_life",
    "seyahat": "culture_life",
    "otomobil": "culture_life",
    "spor": "sports",
    "magazin": "magazine_entertainment",
    "kelebek": "magazine_entertainment",
}

_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid"}


def validate_category(category: str) -> str:
    """Return *category* if it is canonical, otherwise raise ValueError."""
    if category not in CANONICAL_CATEGORIES:
        raise ValueError(
            f"Unknown canonical category {category!r}. "
            f"Expected one of: {', '.join(CANONICAL_CATEGORIES)}"
        )
    return category


def validate_discovery_role(role: str) -> str:
    """Return *role* if it is supported, otherwise raise ValueError."""
    if role not in DISCOVERY_ROLES:
        raise ValueError(
            f"Unknown discovery role {role!r}. "
            f"Expected one of: {', '.join(DISCOVERY_ROLES)}"
        )
    return role


def infer_category_from_url(url: str, default: str = "other") -> str:
    """Infer a canonical category from URL path tokens.

    This is intentionally conservative. General feeds stay ``other`` and are
    separately marked as ``general_discovery`` by ``infer_discovery_role``.
    """
    parts = [
        token
        for part in urlsplit(url).path.lower().replace("_", "-").split("/")
        for token in part.replace(".", "-").split("-")
        if token
    ]
    compact_parts = [part.lower().strip("/") for part in urlsplit(url).path.split("/") if part]

    for token in parts + compact_parts:
        if token in _CATEGORY_TOKEN_MAP:
            return _CATEGORY_TOKEN_MAP[token]
        for keyword, category in _CATEGORY_TOKEN_MAP.items():
            if keyword in token:
                return category
    return validate_category(default)


def infer_discovery_role(url: str) -> str:
    """Classify broad feeds as general discovery, category feeds as topical."""
    path = urlsplit(url).path.lower().strip("/")
    if not path:
        return GENERAL_DISCOVERY

    filename = path.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    if path in {"rss", "feed/rss", "feeds/rss"}:
        return GENERAL_DISCOVERY
    if any(marker in stem or marker in path for marker in ("sondakika", "son-dakika", "anasayfa", "manset")):
        return GENERAL_DISCOVERY

    if infer_category_from_url(url) != "other":
        return CATEGORY_DISCOVERY
    tokens = {path, filename, stem}
    tokens.update(token for token in path.replace("_", "-").replace(".", "-").split("-") if token)

    if tokens & _GENERAL_TOKENS:
        return GENERAL_DISCOVERY
    return CATEGORY_DISCOVERY


def normalize_article_url(url: str) -> str:
    """Canonicalize article URLs for discovery-level deduplication."""
    split = urlsplit((url or "").strip())
    if not split.scheme or not split.netloc:
        return (url or "").strip()

    query_pairs = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        lower = key.lower()
        if lower in _TRACKING_QUERY_KEYS or lower.startswith(_TRACKING_QUERY_PREFIXES):
            continue
        query_pairs.append((key, value))

    path = split.path.rstrip("/") or "/"
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            path,
            query,
            "",
        )
    )
