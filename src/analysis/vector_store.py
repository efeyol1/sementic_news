"""Vector store for semantic news search.

Embeds analyzed news items using sentence-transformers and upserts them
into a ChromaDB collection. Provides similarity search used by the API.

Usage:
    python -m src.analysis.vector_store               # index today's items
    python -m src.analysis.vector_store --date 2026-04-22
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.db.queries import fetch_for_indexing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_COLLECTION_NAME = "news"

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL)
    return _embed_model


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHROMA_DIR = _REPO_ROOT / "data" / "chroma"

# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------


def get_client() -> chromadb.PersistentClient:
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_CHROMA_DIR))


def get_collection(client: chromadb.PersistentClient | None = None) -> chromadb.Collection:
    if client is None:
        client = get_client()
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _build_text(item: dict[str, Any]) -> str:
    title = item.get("cleaned_title", "") or item.get("title", "") or ""
    summary = item.get("cleaned_summary", "") or ""
    return f"{title}. {summary}".strip()


def _item_id(item: dict[str, Any], date_str: str, idx: int) -> str:
    source = (item.get("source_name", "unknown") or "unknown").replace(" ", "_")
    return f"{date_str}_{source}_{idx}"


def index_date(date_str: str) -> int:
    """Embed and upsert all analyzed items for *date_str*. Returns count upserted."""
    items = fetch_for_indexing(date_str)
    if not items:
        logger.warning(f"No items to index for {date_str}")
        return 0

    logger.info(f"Embedding {len(items)} items for {date_str}...")
    model = _get_embed_model()
    texts = [_build_text(i) for i in items]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    collection = get_collection()

    ids = [_item_id(item, date_str, idx) for idx, item in enumerate(items)]
    metadatas = [
        {
            "date": date_str,
            "title": item.get("title", "") or "",
            "source_name": item.get("source_name", "") or "",
            "sentiment_label": item.get("sentiment_label", "") or "",
            "sentiment_score": float(item.get("sentiment_score") or 0),
            "cluster_id": int(item.get("cluster_id") or -1),
            "link": item.get("link", "") or "",
        }
        for item in items
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.success(f"Upserted {len(ids)} items into ChromaDB for {date_str}")
    return len(ids)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def find_similar(
    text: str,
    n: int = 5,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return top-n semantically similar news items to *text*."""
    model = _get_embed_model()
    embedding = model.encode([text])[0].tolist()

    collection = get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n + (1 if exclude_id else 0),
        include=["metadatas", "documents", "distances"],
    )

    output = []
    for meta, doc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        if exclude_id and meta.get("id") == exclude_id:
            continue
        output.append({
            **meta,
            "text": doc,
            "similarity": round(1 - dist, 4),
        })

    return output[:n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index analyzed news into ChromaDB")
    p.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    count = index_date(args.date)
    logger.info(f"Done — {count} items indexed")
    sys.exit(0)
