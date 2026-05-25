"""Entity resolution backfill for ``entity_mentions``.

Local aliases are authoritative and run first. Wikidata linking is optional and
fail-soft; unresolved mentions keep a display canonical without a QID.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

from loguru import logger

from src.analysis.entity_canonicalization import (
    CanonicalEntity,
    canonicalize_mention,
    normalize_entity_text,
)
from src.config import load_country_config
from src.db.queries import (
    bulk_update_entity_resolution,
    fetch_entity_resolution_cache,
    fetch_unresolved_entity_mentions,
    upsert_entity_resolution_cache,
)

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Substring stems matched against the (normalized, accent-stripped, lowercased)
# Wikidata description. Keep them as cross-language stems where one form
# covers several languages: ``politi`` matches politician / politique /
# Politiker(in) / politico / polityk; ``presi`` matches president / présidente
# / presidente; etc. Stricter than full words because checks are substring
# (``keyword in description``) — false positives from wholly unrelated
# descriptions are rare for news entities.
_TYPE_KEYWORDS = {
    "PER": (
        # English
        "human", "person", "politician", "president", "chancellor",
        "minister", "leader", "footballer", "actor", "actress", "athlete",
        "journalist", "musician", "singer", "director", "writer",
        # Cross-language stems (substring, case-insensitive after normalize)
        "politi",   # politician, politique, Politiker(in), politico, polityk
        "presi",    # president, présidente, presidente
        "prasi",    # Präsident (DE, after umlaut strip)
        "prezy",    # prezydent (PL)
        "minist",   # minister, ministre, ministro
        "premier",  # premier ministre, premierminister
        "kanzler",  # Kanzler/Kanzlerin, Bundeskanzler(in)
        "homme d",  # homme d'État / homme d'affaires (FR)
        "femme d",  # femme d'État (FR)
        "chanteur", # singer (FR)
        "chanteuse",
        "acteur", "actrice",            # FR actor/actress
        "schauspieler",                 # DE actor
        "sportler", "spieler",          # DE athlete/player
        "calciatore",                   # IT footballer
        "futbolista",                   # ES footballer
        "deputat", "depute", "deputé",  # deputy/député
        "diplomat",
        "ambassad",  # ambassador, ambassadeur, ambasciatore, ambasador
        "general", "admiral",
        "konig", "konigin",  # König / Königin (DE)
        "roi", "reine",      # FR king/queen
        # Sprint 3 (2026-05-23): expand stems to capture niche IT/ES
        # entities that were scoring 0.55-0.60 against an unrelated description.
        "tennista",                       # IT tennis player (Đoković)
        "scrittor",                       # IT/ES writer (scrittore, escritor)
        "filosof",                        # FR/IT/ES philosopher (Confucio)
        "atleta", "atlet",                # athlete
        "giornalist",                     # IT journalist
        "periodist",                      # ES journalist
        "cantante", "cantantes",          # IT/ES singer
        "musicist", "musico",             # musician
        "regista", "regissor",            # director (IT/ES/PT)
        "filosof",                        # philosopher
        "imprenditor", "empresari",       # IT/ES businessperson
        "rey", "reina",                   # ES king/queen
        "principe", "principessa",        # IT prince/princess
        # Sprint 4 (2026-05-25): EN-specific PER stems for UK rollout.
        # Avoided "king" / "mp" / "sir" / "dame" — substring collisions with
        # "looking" / "campaign" / "desire" / "Notre-Dame" pollute scoring.
        "prime minister", "lord", "queen", "duchess", "duke",
    ),
    "ORG": (
        # English
        "organization", "company", "party", "institution", "agency",
        "association", "club", "league", "team", "bank", "media", "news",
        # Cross-language
        "organisation",                   # FR/DE alt spelling
        "unternehmen", "konzern", "firma",  # DE company
        "entreprise", "societe",            # FR (société → societe)
        "azienda", "compagnia",             # IT
        "empresa",                          # ES
        "partei",                           # DE party
        "parti",                            # FR party
        "partito", "partido",               # IT/ES party
        "partia",                           # PL party
        "verein", "verband",                # DE association
        "agentur",                          # DE agency
        "agence", "agenzia", "agencia", "agencja",
        "institut",                         # institut(e)/Institut(ion)
        "stiftung",                         # DE foundation
        "ministerium",                      # DE ministry
        "klub",                             # PL/TR
        "liga",                             # DE/IT/ES/PL/TR league
        "mannschaft", "squadra", "equipe",  # team
        # Sprint 3 (2026-05-23): government + corporate + religious stems.
        "ministero", "ministerio",          # IT/ES ministry
        "governo", "gobierno",              # IT/ES government
        "parlamento", "parlament",          # parliament
        "asamblea", "assemblea",            # assembly
        "corte", "tribunale", "tribunal",   # court (IT/ES/FR)
        "palazzo", "palacio",               # palace (used for parliament buildings)
        "banca", "banco",                   # bank (IT/ES)
        "fundacao", "fundacion",            # foundation (PT/ES)
        "associazione", "asociacion",       # association (IT/ES)
        "università", "universita", "universidad", "universite",  # university
        "iglesia", "chiesa",                # church
        "procura",                          # IT prosecutor's office
        "fondazione",                       # IT foundation
        # Sprint 4 (2026-05-25): EN-specific ORG stems for UK rollout.
        # Avoided "inc" — substring collisions with "include" / "incident"
        # would over-trigger. "plc" / "ltd" are short but distinctive enough
        # to only appear inside company descriptions.
        "broadcaster", "broadcasting", "corporation",
        "plc", "ltd", "network", "trust", "foundation", "charity",
        "regulator", "watchdog",
    ),
    "LOC": (
        # English
        "city", "country", "state", "municipality", "place", "region",
        "province", "town", "village", "island", "continent", "capital",
        "district", "river", "mountain",
        # Cross-language
        "stadt", "ville", "ciudad", "citta",     # city
        "land", "pays", "paese", "pais", "kraj", "ulke",  # country
        "staat", "estado", "stato", "etat",      # state (état → etat)
        "departement", "department",
        "region", "regione",                     # région → region after strip
        "gemeinde", "commune", "comune", "municipio",
        "dorf",                                  # DE village
        "insel", "ile", "isola", "isla", "wyspa",  # island
        "ort", "lieu", "luogo", "lugar", "miejsce",  # place
        "outre mer", "ubersee",                  # FR/DE overseas
        # Sprint 3 (2026-05-23): lake/river/coast + street stems for IT/ES news.
        "lago", "lac", "see",                    # lake (IT/ES/FR/DE)
        "fiume", "rio", "riviere", "fluss",      # river
        "monte", "montagne", "berg",             # mountain
        "mare", "mer", "meer",                   # sea
        "oceano", "ocean", "ozean",              # ocean
        "calle", "via", "strada", "rue",         # street
        "plaza", "piazza", "place",              # square
        "barrio", "quartier", "quartiere",       # neighborhood
        "comuna", "comuni",                      # municipality (ES/IT plural)
        "metropoli",                             # metropolis
    ),
}

# Wikimedia "polite client" policy: stay at/under ~1 req/s and use a UA
# that identifies the project + contact. Stricter than that and the API
# returns HTTP 429 (observed on first FR/DE live runs on 2026-05-20,
# where 156 sequential requests with no throttle exhausted the quota).
_WIKIDATA_USER_AGENT = (
    "semantic-news/0.2 "
    "(https://github.com/efeyol1/sementic_news; efeyol11@gmail.com)"
)
_WIKIDATA_MIN_INTERVAL = 1.1
_WIKIDATA_MAX_RETRIES = 3
_WIKIDATA_BACKOFF_BASE = 2.0

_last_wikidata_call: float = float("-inf")


def _wikidata_throttle() -> None:
    """Block until ``_WIKIDATA_MIN_INTERVAL`` has elapsed since the last call."""
    global _last_wikidata_call
    now = time.monotonic()
    elapsed = now - _last_wikidata_call
    if elapsed < _WIKIDATA_MIN_INTERVAL:
        time.sleep(_WIKIDATA_MIN_INTERVAL - elapsed)
    _last_wikidata_call = time.monotonic()


def _wikidata_request(url: str, *, timeout: float) -> dict[str, Any]:
    """Throttled GET against the Wikidata API with 429 exponential backoff.

    Raises the underlying ``HTTPError`` if retries are exhausted; the caller
    in ``_resolve_one`` already wraps Wikidata failures in a fail-soft try.
    """
    backoff = _WIKIDATA_BACKOFF_BASE
    last_exc: Exception | None = None
    for attempt in range(_WIKIDATA_MAX_RETRIES + 1):
        _wikidata_throttle()
        req = urllib.request.Request(
            url, headers={"User-Agent": _WIKIDATA_USER_AGENT}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 and attempt < _WIKIDATA_MAX_RETRIES:
                logger.warning(
                    f"Wikidata 429 — backoff {backoff:.1f}s "
                    f"(attempt {attempt + 1}/{_WIKIDATA_MAX_RETRIES})"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    assert last_exc is not None
    raise last_exc


def resolve_entities_batch(
    date_str: str | None = None,
    country_config: dict[str, Any] | None = None,
    limit: int = 1000,
) -> int:
    if country_config is None:
        raise ValueError("country_config is required for entity resolution")
    date_str = date_str or date.today().isoformat()
    country_code = country_config["country_code"]
    cfg = country_config.get("entity_narrative") or {}
    wikidata_cfg = cfg.get("wikidata") or {}
    wikidata_enabled = bool(wikidata_cfg.get("enabled", False))
    min_confidence = float(wikidata_cfg.get("min_confidence", 0.85))

    mentions = fetch_unresolved_entity_mentions(
        date_str,
        country_code=country_code,
        limit=limit,
        include_normalized=wikidata_enabled,
    )
    if not mentions:
        logger.info(f"No unresolved entity mentions for {date_str} [{country_code}]")
        return 0

    updates: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    for mention in mentions:
        result = _resolve_one(
            mention,
            country_config=country_config,
            wikidata_enabled=wikidata_enabled,
            min_confidence=min_confidence,
        )
        updates.append(
            {
                "id": mention["id"],
                "canonical": result.canonical,
                "wikidata_qid": result.wikidata_qid,
                "resolution_confidence": result.confidence,
                "resolver_method": result.resolver_method,
            }
        )
        cache_rows.append(
            {
                "normalized_text": normalize_entity_text(
                    mention["entity_text"],
                    country_config.get("language"),
                ),
                "entity_type": mention["entity_type"],
                "country_code": country_code,
                "canonical": result.canonical,
                "wikidata_qid": result.wikidata_qid,
                "confidence": result.confidence,
                "resolver_method": result.resolver_method,
            }
        )

    bulk_update_entity_resolution(updates)
    upsert_entity_resolution_cache(cache_rows)
    logger.info(
        f"entity_resolution done — {len(updates)} mentions for {date_str} "
        f"[{country_code}], wikidata_enabled={wikidata_enabled}"
    )
    return len(updates)


def _resolve_one(
    mention: dict[str, Any],
    country_config: dict[str, Any],
    wikidata_enabled: bool,
    min_confidence: float,
) -> CanonicalEntity:
    local = canonicalize_mention(
        mention["entity_text"],
        mention["entity_type"],
        country_config,
    )
    if local.resolver_method == "local_alias":
        return local

    normalized = normalize_entity_text(mention["entity_text"], country_config.get("language"))
    cached = fetch_entity_resolution_cache(
        normalized,
        mention["entity_type"],
        country_config["country_code"],
    )
    if cached:
        return CanonicalEntity(
            canonical=cached.get("canonical") or local.canonical,
            wikidata_qid=cached.get("wikidata_qid"),
            confidence=float(cached.get("confidence") or 0.0),
            resolver_method=cached.get("resolver_method") or "cache",
        )

    if not wikidata_enabled:
        return local

    try:
        wikidata = _resolve_wikidata(
            mention["entity_text"],
            mention["entity_type"],
            country_config,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        logger.warning(f"Wikidata entity resolution failed soft for {mention['entity_text']!r}: {exc}")
        return local
    return wikidata or local


def _resolve_wikidata(
    text: str,
    entity_type: str,
    country_config: dict[str, Any],
    min_confidence: float,
    timeout: float = 5.0,
) -> CanonicalEntity | None:
    language = country_config.get("language") or "en"
    params = urllib.parse.urlencode(
        {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "uselang": language,
            "limit": 5,
            "search": text,
        }
    )
    payload = _wikidata_request(f"{_WIKIDATA_API}?{params}", timeout=timeout)

    candidates = payload.get("search") or []
    if not candidates:
        return None

    scored = [
        (_score_candidate(candidate, text, entity_type, country_config), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    score, candidate = scored[0]
    if score < min_confidence:
        return None
    return CanonicalEntity(
        canonical=candidate.get("label") or text,
        wikidata_qid=candidate.get("id"),
        confidence=round(score, 4),
        resolver_method="wikidata",
    )


def _score_candidate(
    candidate: dict[str, Any],
    text: str,
    entity_type: str,
    country_config: dict[str, Any],
) -> float:
    language = country_config.get("language")
    normalized_text = normalize_entity_text(text, language)
    label = normalize_entity_text(candidate.get("label") or "", language)
    description = normalize_entity_text(candidate.get("description") or "", language)
    aliases = [
        normalize_entity_text(alias, language)
        for alias in (candidate.get("aliases") or [])
        if alias
    ]

    # Hard reject Wikidata disambiguation pages — surname / cognome /
    # apellido / family-name entries. They share a label with real
    # entities ("Djokovic" → Q21146583 cognome before Q5812 Novak; same
    # for "Rufián", "Confucio") and otherwise pass threshold. The
    # description is the reliable signal: these pages explicitly
    # self-identify as a surname / family name in the local language.
    disambig_stems = (
        "apellido",       # ES surname
        "cognome",        # IT surname
        "surname", "family name",  # EN
        "nazwisko",       # PL surname
        "nachname",       # DE surname
        "nom de famille", # FR family name
        "page d homonymie", "pagina di disambiguazione",  # disambig page (FR/IT)
        "disambiguation", "disambiguacion", "begriffsklarung",  # disambig EN/ES/DE
    )
    if any(stem in description for stem in disambig_stems):
        return 0.0

    score = 0.0
    if label == normalized_text:
        score += 0.55
    elif normalized_text in aliases:
        # Sprint 4 (2026-05-25): alias match weight 0.50 → 0.55 to match
        # exact-label weight. Wikidata aliases are curated equivalents
        # (e.g. "Boris Johnson" alias of "Alexander Boris de Pfeffel Johnson"),
        # so penalising them under label match was the wrong default.
        score += 0.55
    elif label and (label in normalized_text or normalized_text in label):
        score += 0.35

    type_keywords = _TYPE_KEYWORDS.get(str(entity_type).upper(), ())
    if any(keyword in description for keyword in type_keywords):
        score += 0.15

    country_terms = {
        normalize_entity_text(country_config.get("country_name") or "", language),
        normalize_entity_text(country_config.get("country_slug") or "", language),
        normalize_entity_text(country_config.get("country_code") or "", language),
    }
    if any(term and term in description for term in country_terms):
        score += 0.10

    if candidate.get("id"):
        score += 0.05
    return min(1.0, score)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve entity_mentions to canonical names/QIDs.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--country", required=True, help="Country slug or ISO code")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_country_config(args.country)
    resolve_entities_batch(date_str=args.date, country_config=cfg, limit=args.limit)
    sys.exit(0)
