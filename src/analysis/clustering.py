"""Topic clustering for Turkish news items.

Reads items from PostgreSQL, clusters by topic using sentence-transformer
embeddings + KMeans, and writes enriched results back to the same rows.

Each item gains:
    cluster_id       int   — cluster index (0-based)
    cluster_keywords list  — top TF-IDF terms that define the cluster
    cluster_title    str   — human-readable cluster name

A cluster summary is also written to the cluster_summaries table.

Usage:
    python -m src.analysis.clustering               # cluster today's items
    python -m src.analysis.clustering --date 2026-04-17
    python -m src.analysis.clustering --n-clusters 20
"""

import argparse
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.db.queries import (
    bulk_update_clustering,
    fetch_for_clustering,
    upsert_cluster_summaries,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_CLUSTERS = 15
TOP_KEYWORDS_PER_CLUSTER = 8

_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TURKISH_STOPWORDS = [
    "bir", "bu", "ve", "ile", "için", "de", "da", "den", "dan", "mi",
    "mı", "mu", "mü", "ne", "o", "ya", "ki", "ama", "en", "çok", "daha",
    "olan", "oldu", "olarak", "olan", "var", "yok", "gibi", "kadar",
    "sonra", "önce", "her", "biz", "siz", "onlar", "ben", "sen",
    "şu", "hangi", "nasıl", "neden", "çünkü", "ancak", "fakat",
    "hem", "veya", "yani", "ise", "iken", "diye", "göre",
    "üzere", "karşı", "doğru", "içinde", "üzerinde", "altında",
    "türkiye", "türk", "yıl", "gün", "ay", "saat", "kişi", "kez",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
    "peki", "zaman", "artık", "sadece", "bile", "hiç", "çünkü",
    "olacak", "olan", "oldu", "olup", "olmak", "olmadan",
    "şimdi", "geçen", "geldi", "gelecek", "yapılan", "yapıldı",
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
    "2024", "2025", "2026", "son", "yeni", "büyük", "ilk", "önemli",
]

mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-clustering")

# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def _build_corpus(items: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    indices, texts = [], []
    for i, item in enumerate(items):
        if not item.get("is_turkish", True):
            continue
        title = item.get("cleaned_title", "") or ""
        summary = item.get("cleaned_summary", "") or ""
        text = f"{title} {summary}".strip()
        if len(text) >= 20:
            indices.append(i)
            texts.append(text)
    return indices, texts


def _embed(texts: list[str]) -> np.ndarray:
    logger.info(f"Encoding {len(texts)} texts with {_EMBED_MODEL}...")
    model = SentenceTransformer(_EMBED_MODEL)
    return model.encode(texts, batch_size=64, show_progress_bar=True)


def _cluster(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    n_clusters = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def _top_keywords(
    cluster_id: int,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int,
) -> list[str]:
    mask = labels == cluster_id
    if not mask.any():
        return []
    centroid = np.asarray(matrix[mask].mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    top_indices = centroid.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices]


def _generate_title(keywords: list[str], cluster_items: list[dict]) -> str:
    entity_counter: Counter = Counter()
    for item in cluster_items:
        for label in ("ORG", "LOC", "PER"):
            ents = (item.get("entities") or {}).get(label, [])
            entity_counter.update(ents)

    top_keyword = keywords[0].title() if keywords else "Genel"

    if entity_counter:
        top_entity = entity_counter.most_common(1)[0][0]
        if top_entity.lower() != top_keyword.lower():
            return f"{top_keyword} · {top_entity}"

    return top_keyword


# ---------------------------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------------------------


def _build_cluster_summaries(
    items: list[dict[str, Any]],
    indices: list[int],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    n_clusters: int,
) -> list[dict[str, Any]]:
    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for pos, item_idx in enumerate(indices):
        cluster_indices[int(labels[pos])].append(item_idx)

    summaries = []
    for cid in range(n_clusters):
        item_idxs = cluster_indices.get(cid, [])
        cluster_items = [items[i] for i in item_idxs]
        keywords = _top_keywords(cid, labels, vectorizer, matrix, TOP_KEYWORDS_PER_CLUSTER)
        title = _generate_title(keywords, cluster_items)
        sources = Counter(items[i]["source_name"] for i in item_idxs)
        sentiments = Counter(
            items[i].get("sentiment_label") for i in item_idxs
            if items[i].get("sentiment_label")
        )
        summaries.append({
            "cluster_id": cid,
            "title": title,
            "size": len(item_idxs),
            "keywords": keywords,
            "sources": dict(sources.most_common()),
            "sentiment_distribution": dict(sentiments),
        })

    summaries.sort(key=lambda x: x["size"], reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_topics(
    date_str: str | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> int:
    """Run topic clustering for a single day's items.

    Returns:
        Number of items clustered.
    """
    date_str = date_str or date.today().isoformat()

    items = fetch_for_clustering(date_str)
    if not items:
        logger.warning(f"No items found for clustering on {date_str}")
        return 0

    t0 = time.perf_counter()
    indices, texts = _build_corpus(items)

    if len(texts) < n_clusters:
        logger.warning(f"Only {len(texts)} usable items — reducing n_clusters to {len(texts)}")
        n_clusters = max(2, len(texts))

    embeddings = _embed(texts)
    labels = _cluster(embeddings, n_clusters)

    sil_score = float(silhouette_score(embeddings, labels, metric="cosine"))
    duration = time.perf_counter() - t0
    logger.info(f"Clustering done — silhouette={sil_score:.4f} in {duration:.1f}s")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=_TURKISH_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    summaries = _build_cluster_summaries(items, indices, labels, vectorizer, matrix, n_clusters)
    title_map = {s["cluster_id"]: s["title"] for s in summaries}

    # Build per-item cluster update list
    cluster_updates: list[dict[str, Any]] = []
    for cid in range(n_clusters):
        kws = _top_keywords(cid, labels, vectorizer, matrix, TOP_KEYWORDS_PER_CLUSTER)
        for pos, item_idx in enumerate(indices):
            if labels[pos] == cid:
                cluster_updates.append({
                    "id": items[item_idx]["id"],
                    "cluster_id": cid,
                    "cluster_keywords": kws,
                    "cluster_title": title_map.get(cid, ""),
                })

    bulk_update_clustering(cluster_updates)
    upsert_cluster_summaries(summaries, date_str)

    with mlflow.start_run(run_name=f"clustering-{date_str}"):
        mlflow.log_params({
            "date": date_str,
            "n_clusters": n_clusters,
            "corpus_size": len(texts),
            "embed_model": _EMBED_MODEL,
        })
        mlflow.log_metrics({
            "silhouette_score": sil_score,
            "duration_sec": duration,
            "avg_cluster_size": len(texts) / n_clusters,
        })

    logger.info("Top clusters:")
    for s in summaries[:5]:
        logger.info(f"  [{s['cluster_id']}] {s['title']} ({s['size']} items): {', '.join(s['keywords'][:4])}")

    return len(cluster_updates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster Turkish news items by topic.")
    parser.add_argument("--date", default=date.today().isoformat(), metavar="YYYY-MM-DD")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS, metavar="N")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cluster_topics(date_str=args.date, n_clusters=args.n_clusters)
    sys.exit(0)
