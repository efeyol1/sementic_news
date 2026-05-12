"""Vector indexing and semantic search using pgvector.

Embeds analyzed news items with sentence-transformers and stores the vectors
in the news_items.embedding column (PostgreSQL + pgvector extension).

Usage:
    python -m src.analysis.vector_store               # index today's items
    python -m src.analysis.vector_store --date 2026-04-22
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any

from loguru import logger
from sentence_transformers import SentenceTransformer

from src.analysis.text_inputs import build_embedding_text
from src.db.queries import (
    bulk_update_embeddings,
    fetch_for_indexing,
    find_similar_pgvector,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL)
    return _embed_model


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _build_text(item: dict[str, Any]) -> str:
    return build_embedding_text(item)


def index_date(date_str: str, country_code: str = "TR") -> int:
    """Embed all analyzed items for *date_str* and store in PostgreSQL.

    Returns count of items indexed.
    """
    items = fetch_for_indexing(date_str, country_code=country_code)
    if not items:
        logger.warning(f"No items to index for {date_str}")
        return 0

    logger.info(f"Embedding {len(items)} items for {date_str}...")
    model = _get_embed_model()
    texts = [_build_text(i) for i in items]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    updates = [
        {"id": item["id"], "embedding": embeddings[i].tolist()}
        for i, item in enumerate(items)
    ]

    bulk_update_embeddings(updates)
    logger.success(f"Indexed {len(updates)} items for {date_str}")
    return len(updates)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def find_similar(text: str, n: int = 5, country_code: str = "TR") -> list[dict[str, Any]]:
    """Return top-n semantically similar news items to *text*."""
    model = _get_embed_model()
    embedding = model.encode([text])[0].tolist()
    results = find_similar_pgvector(embedding, n=n, country_code=country_code)
    for r in results:
        r["similarity"] = round(float(r["similarity"]), 4)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index analyzed news into pgvector")
    p.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    count = index_date(args.date)
    logger.info(f"Done — {count} items indexed")
    sys.exit(0)
