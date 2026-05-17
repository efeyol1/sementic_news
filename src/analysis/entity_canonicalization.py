"""Deterministic entity canonicalization helpers.

This layer is intentionally boring: it normalizes mention text and applies
country-configured aliases before any external entity linker runs. The output
is shaped like a resolver result so the later Wikidata/entity-registry path can
reuse the same fields without changing callers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_TITLE_PREFIXES = {
    "tr": {
        "cumhurbaskani",
        "baskan",
        "bakan",
        "basbakan",
        "prof",
        "dr",
        "sayin",
    },
    "de": {
        "bundeskanzler",
        "bundeskanzlerin",
        "kanzler",
        "kanzlerin",
        "prasident",
        "president",
        "minister",
        "dr",
    },
    "en": {
        "president",
        "prime minister",
        "minister",
        "chancellor",
        "senator",
        "mr",
        "mrs",
        "ms",
        "dr",
    },
}


@dataclass(frozen=True)
class CanonicalEntity:
    canonical: str
    wikidata_qid: str | None = None
    confidence: float = 0.0
    resolver_method: str = "normalized"


def normalize_entity_text(text: str, language: str | None = None) -> str:
    """Fold entity text for deterministic alias lookups."""
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("ı", "i")
    value = _PUNCT_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    value = _strip_title_prefixes(value, language=language)
    return value


def display_canonical(text: str, language: str | None = None) -> str:
    """Clean a raw mention for display when no configured alias matches."""
    normalized = normalize_entity_text(text, language=language)
    if not normalized:
        return str(text or "").strip()
    words = [
        token.upper() if len(token) <= 3 and token.isascii() else token.capitalize()
        for token in normalized.split()
    ]
    return " ".join(words)


def canonicalize_mention(
    entity_text: str,
    entity_type: str,
    country_config: dict[str, Any],
) -> CanonicalEntity:
    """Return the best local canonical form for one extracted mention."""
    language = country_config.get("language")
    alias_index = build_alias_index(country_config)
    key = (str(entity_type or "").upper(), normalize_entity_text(entity_text, language))
    if key in alias_index:
        return alias_index[key]
    return CanonicalEntity(canonical=display_canonical(entity_text, language=language))


def build_alias_index(country_config: dict[str, Any]) -> dict[tuple[str, str], CanonicalEntity]:
    cfg = country_config.get("entity_narrative") or {}
    aliases_cfg = cfg.get("aliases") or {}
    language = country_config.get("language")
    index: dict[tuple[str, str], CanonicalEntity] = {}

    for bucket, entity_type in (("people", "PER"), ("organizations", "ORG")):
        for canonical, payload in (aliases_cfg.get(bucket) or {}).items():
            aliases, qid = _parse_alias_payload(canonical, payload)
            for alias in {canonical, *aliases}:
                normalized = normalize_entity_text(alias, language)
                if not normalized:
                    continue
                index[(entity_type, normalized)] = CanonicalEntity(
                    canonical=str(canonical),
                    wikidata_qid=qid,
                    confidence=1.0,
                    resolver_method="local_alias",
                )
    return index


def _parse_alias_payload(canonical: str, payload: Any) -> tuple[set[str], str | None]:
    if isinstance(payload, dict):
        raw_aliases = payload.get("aliases") or []
        qid = payload.get("wikidata_qid")
    elif isinstance(payload, (list, tuple, set)):
        raw_aliases = payload
        qid = None
    elif payload is None:
        raw_aliases = []
        qid = None
    else:
        raw_aliases = [payload]
        qid = None
    aliases = {str(alias) for alias in raw_aliases}
    aliases.add(str(canonical))
    return aliases, str(qid) if qid else None


def _strip_title_prefixes(value: str, language: str | None = None) -> str:
    prefixes = set(_TITLE_PREFIXES.get(language or "", set()))
    prefixes.update(_TITLE_PREFIXES["en"])
    prefixes.update(_TITLE_PREFIXES["tr"])
    prefixes.update(_TITLE_PREFIXES["de"])

    changed = True
    while changed and value:
        changed = False
        for prefix in sorted(prefixes, key=len, reverse=True):
            if value == prefix:
                return ""
            if value.startswith(prefix + " "):
                value = value[len(prefix) + 1 :].strip()
                changed = True
                break
    return value
