"""Entity collocation extraction (Sprint 5 of the entity-narrative track).

For each entity mention recorded in ``entity_mentions``, take a window
of ±N tokens around the mention, lemmatize, filter by POS allowlist +
stopwords + minimum length, and aggregate into per-day
``(country_code, collected_date, canonical, entity_type, lemma, pos)``
cooccurrence counts written to ``entity_collocations``.

The mention's ``position_in_article`` is a character offset into the
exact string the HuggingFace NER pipeline consumed, namely
:func:`src.analysis.text_inputs.build_ner_text`'s output
(``title + ". " + summary + ". " + body[:3000]``). The lemmatizer is
fed *the same* reconstructed string so token char-spans line up with
mention positions. A pinned test (``test_position_matches_build_ner_text``)
guards against drift.

Sprint 6 will fill ``pmi`` / ``log_likelihood`` / the four ``*_total``
columns this migration already creates. Sprint 5 only writes raw counts.

Usage:
    python -m src.analysis.collocations --country germany --date 2026-05-25
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date
from typing import Any

from loguru import logger

from src.analysis.text_inputs import build_ner_text
from src.analysis.text_processing import (
    Token,
    filter_tokens,
    get_lemmatizer,
    load_stopwords,
)
from src.config import load_country_config
from src.db.queries import (
    bulk_upsert_entity_collocations,
    fetch_for_collocation_extraction,
)

# ---------------------------------------------------------------------------
# Defaults — overridable per country YAML
# ---------------------------------------------------------------------------


DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "window": 10,
    "respect_sentence_boundary": True,
    "keep_pos": ["NOUN", "PROPN", "VERB", "ADJ"],
    "min_lemma_length": 3,
    "min_cooccurrence_count": 2,
    "lemmatizer": None,        # auto-pick from country language
    "spacy_model": None,       # auto-pick from language defaults
    "stopwords_file": None,    # required if you want stopword filtering
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _collocations_cfg(country_config: dict[str, Any]) -> dict[str, Any]:
    return ((country_config.get("entity_narrative") or {})
            .get("collocations") or {})


def is_enabled(country_config: dict[str, Any]) -> bool:
    return bool(_collocations_cfg(country_config).get("enabled", DEFAULTS["enabled"]))


def _setting(country_config: dict[str, Any], key: str) -> Any:
    cfg = _collocations_cfg(country_config)
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return DEFAULTS.get(key)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------


def _find_mention_span(
    tokens: list[Token], position: int, entity_text: str
) -> tuple[int, int]:
    """Return ``(first_idx, last_idx)`` of tokens overlapping the mention.

    ``position`` is the character offset where the entity starts in the
    string the lemmatizer received (must be the same string the NER
    pipeline received — ``build_ner_text(item)``). Returns ``(-1, -1)``
    if no token falls in the span.
    """
    entity_end = position + len(entity_text)
    first = -1
    last = -1
    for i, tok in enumerate(tokens):
        if tok.end <= position:
            continue
        if tok.start >= entity_end:
            break
        if first == -1:
            first = i
        last = i
    return first, last


def _window_tokens(
    tokens: list[Token],
    first_idx: int,
    last_idx: int,
    window: int,
    respect_sentence: bool,
) -> list[Token]:
    """Return up to ``±window`` tokens around the mention (exclusive of
    the mention's own tokens). Optionally bounded by sentence id."""
    if first_idx < 0:
        return []
    anchor_sent = tokens[first_idx].sent_id
    out: list[Token] = []
    left_start = max(0, first_idx - window)
    for i in range(left_start, first_idx):
        if respect_sentence and tokens[i].sent_id != anchor_sent:
            continue
        out.append(tokens[i])
    right_end = min(len(tokens), last_idx + 1 + window)
    for i in range(last_idx + 1, right_end):
        if respect_sentence and tokens[i].sent_id != anchor_sent:
            continue
        out.append(tokens[i])
    return out


# ---------------------------------------------------------------------------
# Aggregation key
# ---------------------------------------------------------------------------


def _aggregation_key(row: dict[str, Any], lemma: str, pos: str) -> tuple:
    """One row per (canonical, qid, entity_type, lemma, pos) per day."""
    return (
        row["canonical"],
        row.get("wikidata_qid"),
        row["entity_type"],
        lemma,
        pos,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_collocations_batch(
    date_str: str | None = None,
    country_config: dict[str, Any] | None = None,
) -> int:
    """Compute one day's collocation counts and persist them.

    Returns the number of aggregated rows inserted.
    """
    if country_config is None:
        raise ValueError("country_config is required for collocation extraction")
    if not is_enabled(country_config):
        logger.info(
            f"entity_narrative.collocations disabled for "
            f"{country_config.get('country_slug', '?')} — skipping"
        )
        return 0

    date_str = date_str or date.today().isoformat()
    country_code: str = country_config["country_code"]
    language: str = country_config.get("language") or "en"

    rows = fetch_for_collocation_extraction(date_str, country_code=country_code)
    if not rows:
        logger.warning(
            f"No mentions with position found for collocation extraction "
            f"on {date_str} [{country_code}]"
        )
        bulk_upsert_entity_collocations([], country_code, date_str)
        return 0

    # --- Resolve config knobs once -----------------------------------------
    window = int(_setting(country_config, "window"))
    respect_sentence = bool(_setting(country_config, "respect_sentence_boundary"))
    min_lemma_length = int(_setting(country_config, "min_lemma_length"))
    min_cooccurrence = int(_setting(country_config, "min_cooccurrence_count"))
    keep_pos_list = _setting(country_config, "keep_pos") or []
    keep_pos: set[str] | None = {str(p).upper() for p in keep_pos_list} or None

    backend = _setting(country_config, "lemmatizer") or (
        "surface" if language == "tr" else "spacy"
    )
    spacy_model = _setting(country_config, "spacy_model")
    if backend == "surface":
        lemmatizer = get_lemmatizer("tr")
    else:
        lemmatizer = get_lemmatizer(language, model_name=spacy_model)

    stopwords_file = _setting(country_config, "stopwords_file")
    stopwords: set[str] = (
        load_stopwords(stopwords_file) if stopwords_file else set()
    )

    # --- Group mentions by article so we tokenize each article once --------
    by_article: dict[int, list[dict[str, Any]]] = {}
    article_items: dict[int, dict[str, Any]] = {}
    for r in rows:
        by_article.setdefault(r["article_id"], []).append(r)
        # Preserve full text fields keyed by article id; mention rows
        # duplicate them on the JOIN but the per-article values are
        # identical, so caching the first one is correct.
        article_items.setdefault(r["article_id"], r)

    # --- Walk each article, aggregate ---------------------------------------
    t0 = time.perf_counter()
    aggregate: Counter[tuple] = Counter()
    qid_for_key: dict[tuple, str | None] = {}
    skipped_mentions = 0

    for article_id, mentions in by_article.items():
        item = article_items[article_id]
        text = build_ner_text(item)
        if not text:
            skipped_mentions += len(mentions)
            continue
        tokens = lemmatizer(text)
        if not tokens:
            skipped_mentions += len(mentions)
            continue

        for mention in mentions:
            position = mention["position_in_article"]
            entity_text = mention["entity_text"] or ""
            if position is None or position < 0 or position >= len(text):
                skipped_mentions += 1
                continue
            first, last = _find_mention_span(tokens, position, entity_text)
            if first < 0:
                skipped_mentions += 1
                continue
            window_toks = _window_tokens(
                tokens, first, last, window, respect_sentence
            )
            kept = filter_tokens(
                window_toks,
                stopwords=stopwords,
                keep_pos=keep_pos,
                min_lemma_length=min_lemma_length,
            )
            for tok in kept:
                key = _aggregation_key(mention, tok.lemma, tok.pos)
                aggregate[key] += 1
                qid_for_key.setdefault(key, mention.get("wikidata_qid"))

    # --- Threshold + materialize rows --------------------------------------
    output_rows: list[dict[str, Any]] = []
    for key, count in aggregate.items():
        if count < min_cooccurrence:
            continue
        canonical, _qid_in_key, entity_type, lemma, pos = key
        output_rows.append(
            {
                "canonical": canonical,
                "wikidata_qid": qid_for_key.get(key),
                "entity_type": entity_type,
                "lemma": lemma,
                "pos": pos,
                "cooccurrence_count": count,
            }
        )

    inserted = bulk_upsert_entity_collocations(
        output_rows, country_code, date_str
    )
    elapsed = time.perf_counter() - t0
    logger.success(
        f"Collocations [{country_code} {date_str}]: "
        f"{len(by_article)} articles, {len(rows)} mentions, "
        f"{skipped_mentions} skipped, {inserted} aggregated rows "
        f"(min_cooccurrence={min_cooccurrence}) in {elapsed:.1f}s"
    )
    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run entity collocation extraction for one country/day."
    )
    parser.add_argument("--country", required=True, help="country slug or code")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date to process (default: today)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    country_config = load_country_config(args.country)
    extract_collocations_batch(date_str=args.date, country_config=country_config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
