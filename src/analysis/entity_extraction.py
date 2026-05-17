"""Multilingual entity extraction (Sprint 1 of entity-narrative track).

Runs a country-configurable HuggingFace NER model over each day's
news items and writes one row per detected entity mention into the
``entity_mentions`` table. The legacy ``news_items.entities`` JSONB
column populated by :mod:`src.analysis.ner` stays untouched — both
tracks run in parallel until Sprint 9 deprecates the JSONB path.

Model selection (per country YAML):
    entity_narrative:
      enabled: true
      ner_model: Davlan/bert-base-multilingual-cased-ner-hrl
      keep_labels: [PER, ORG, LOC, MISC]

The default model covers 10 languages including DE, FR, IT, ES, PL — a
single artifact for Sprints 1–4. Languages with a clear quality cliff
(measured in Sprint 4) get a per-language override.

Usage:
    python -m src.analysis.entity_extraction --country germany --date 2026-05-14
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from typing import Any

import torch
from loguru import logger
from transformers import pipeline

from src.analysis.entity_canonicalization import canonicalize_mention
from src.analysis.text_inputs import build_ner_text
from src.config import load_country_config
from src.db.queries import bulk_insert_entity_mentions, fetch_for_entity_extraction

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Davlan/bert-base-multilingual-cased-ner-hrl"
DEFAULT_KEEP_LABELS = ("PER", "ORG", "LOC", "MISC")
# Quality filters tuned against the 2026-05-16 DE smoke run:
# ``"simple"`` aggregation leaked WordPiece fragments ("mer", "##i jinping",
# tek harfli "x"/"j"/"al" gibi). ``"average"`` span birleşmesini düzeltir;
# kalan artifact'ler için subword/length/confidence filtrelemesi.
# Score gate ``0.85`` ilk denemede çok yüksek çıktı (Davlan multilingual NER
# Avrupa dillerinde tipik olarak 0.6–0.85 aralığında skorluyor); 417→51
# mention kaybı oldu. Subword + length filtreleri zaten "asıl gürültüyü"
# elediği için score gate'i 0.5'e düşürüyoruz — yalnızca model-çok-emin-değil
# vakalarını filtreliyoruz.
DEFAULT_AGGREGATION_STRATEGY = "average"
# 2 → 3: tek-iki harfli token'lar (örn. "AL", "X", "JI") tokenizer artifact;
# multilingual NER Arapça artikel ``al-`` veya Çince tek-hece soyadları gibi
# vakalarda standalone fragment üretiyor. 3-char gate canonical-resolvable
# isimleri tutar, fragment'leri eler.
MIN_ENTITY_LENGTH = 3
MIN_ENTITY_SCORE = 0.5
_SUBWORD_MARKER = "##"

_PIPELINE_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _entity_narrative_cfg(country_config: dict[str, Any]) -> dict[str, Any]:
    return country_config.get("entity_narrative") or {}


def is_enabled(country_config: dict[str, Any]) -> bool:
    """Step gate. Defaults to ``False`` so configs without the section opt out."""
    return bool(_entity_narrative_cfg(country_config).get("enabled", False))


def _model_id(country_config: dict[str, Any]) -> str:
    return _entity_narrative_cfg(country_config).get("ner_model") or DEFAULT_MODEL_ID


def _keep_labels(country_config: dict[str, Any]) -> set[str]:
    raw = _entity_narrative_cfg(country_config).get("keep_labels") or DEFAULT_KEEP_LABELS
    return {str(label).upper() for label in raw}


# ---------------------------------------------------------------------------
# Model loader (cached so re-runs in the same process don't re-download)
# ---------------------------------------------------------------------------


def _load_pipeline(model_id: str):
    cached = _PIPELINE_CACHE.get(model_id)
    if cached is not None:
        return cached
    device = 0 if torch.cuda.is_available() else -1
    logger.info(
        f"Loading multilingual NER model {model_id!r} on "
        f"{'CUDA' if device == 0 else 'CPU'}"
    )
    pipe = pipeline(
        "ner",
        model=model_id,
        aggregation_strategy=DEFAULT_AGGREGATION_STRATEGY,
        device=device,
    )
    _PIPELINE_CACHE[model_id] = pipe
    logger.success("Multilingual NER model loaded.")
    return pipe


# ---------------------------------------------------------------------------
# Per-article extraction
# ---------------------------------------------------------------------------


def _normalize_label(raw_label: str) -> str:
    """Strip BIO prefix variations the HF pipeline may emit (B-PER, I-ORG, …)."""
    label = raw_label.upper().strip()
    if "-" in label:
        label = label.split("-", 1)[1]
    return label


def _is_subword_artifact(word: str) -> bool:
    """``##wort``, ``##i jinping`` gibi WordPiece sızıntılarını yakalar."""
    if not word:
        return True
    if word.startswith(_SUBWORD_MARKER):
        return True
    # Aggregation sonrası içinde ``##`` kalmış token (örn. ``##i jinping``):
    # boşlukla ayrılmış parçalardan biri marker ile başlıyorsa reddet.
    return any(part.startswith(_SUBWORD_MARKER) for part in word.split())


def _extract_mentions(
    text: str,
    pipe,
    keep_labels: set[str],
) -> list[dict[str, Any]]:
    if not text:
        return []
    raw: list[dict[str, Any]] = pipe(text)
    mentions: list[dict[str, Any]] = []
    for ent in raw:
        label = _normalize_label(ent.get("entity_group") or ent.get("entity") or "")
        if label not in keep_labels:
            continue
        word = (ent.get("word") or "").strip()
        if not word or _is_subword_artifact(word):
            continue
        # Single-character / sub-MIN-length fragments are almost always
        # tokenizer noise ("x", "j", "al", "hen").
        if len(word) < MIN_ENTITY_LENGTH:
            continue
        # ``aggregation_strategy="average"`` populates ``score`` per
        # aggregated span; gate on it so low-confidence guesses don't
        # show up in collocation profiles.
        score = ent.get("score")
        if score is not None and score < MIN_ENTITY_SCORE:
            continue
        mentions.append(
            {
                "entity_text": word,
                "entity_type": label,
                "position_in_article": ent.get("start"),
            }
        )
    return mentions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities_batch(
    date_str: str | None = None,
    country_config: dict[str, Any] | None = None,
) -> int:
    """Run multilingual NER for one day's items and persist mentions.

    Returns:
        Total number of mention rows inserted.
    """
    if country_config is None:
        raise ValueError("country_config is required for entity extraction")
    if not is_enabled(country_config):
        logger.info(
            f"entity_narrative disabled for "
            f"{country_config.get('country_slug', '?')} — skipping"
        )
        return 0

    date_str = date_str or date.today().isoformat()
    country_code: str = country_config["country_code"]

    items = fetch_for_entity_extraction(date_str, country_code=country_code)
    if not items:
        logger.warning(
            f"No items found for entity extraction on {date_str} "
            f"[{country_code}]"
        )
        return 0

    keep_labels = _keep_labels(country_config)
    pipe = _load_pipeline(_model_id(country_config))

    t0 = time.perf_counter()
    all_mentions: list[dict[str, Any]] = []
    article_ids: list[int] = [item["id"] for item in items]
    for i, item in enumerate(items, 1):
        text = build_ner_text(item)
        mentions = _extract_mentions(text, pipe, keep_labels)
        for mention in mentions:
            resolved = canonicalize_mention(
                mention["entity_text"],
                mention["entity_type"],
                country_config,
            )
            mention["article_id"] = item["id"]
            mention["country_code"] = country_code
            mention["collected_date"] = date_str
            mention["canonical"] = resolved.canonical
            mention["wikidata_qid"] = resolved.wikidata_qid
            mention["resolution_confidence"] = resolved.confidence
            mention["resolver_method"] = resolved.resolver_method
        all_mentions.extend(mentions)
        if i % 50 == 0:
            logger.debug(f"entity_extraction progress: {i}/{len(items)}")
    duration = time.perf_counter() - t0

    inserted = bulk_insert_entity_mentions(all_mentions, article_ids)

    per = sum(1 for m in all_mentions if m["entity_type"] == "PER")
    org = sum(1 for m in all_mentions if m["entity_type"] == "ORG")
    loc = sum(1 for m in all_mentions if m["entity_type"] == "LOC")
    misc = sum(1 for m in all_mentions if m["entity_type"] == "MISC")
    logger.info(
        f"entity_extraction done — {len(items)} articles, {inserted} mentions "
        f"(PER={per}, ORG={org}, LOC={loc}, MISC={misc}) in {duration:.1f}s"
    )
    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multilingual entity extraction → entity_mentions table."
    )
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument(
        "--country",
        required=True,
        help="Country slug or ISO code (must have entity_narrative.enabled: true)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_country_config(args.country)
    extract_entities_batch(date_str=args.date, country_config=cfg)
    sys.exit(0)
