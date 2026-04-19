"""Topic clustering for analyzed Turkish news items.

Reads data/analyzed/YYYY-MM-DD.json, clusters items by topic using TF-IDF +
KMeans, and writes enriched results back to the same file.

Each item gains:
    cluster_id       int   — cluster index (0-based)
    cluster_keywords list  — top TF-IDF terms that define the cluster

A cluster summary is also written to data/analyzed/YYYY-MM-DD_clusters.json.

Usage:
    python -m src.analysis.clustering               # cluster today's file
    python -m src.analysis.clustering --date 2026-04-17
    python -m src.analysis.clustering --n-clusters 20
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_CLUSTERS = 15
TOP_KEYWORDS_PER_CLUSTER = 8

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ANALYZED_DIR = _REPO_ROOT / "data" / "analyzed"

# Common Turkish stopwords to exclude from cluster keywords
_TURKISH_STOPWORDS = [
    "bir", "bu", "ve", "ile", "için", "de", "da", "den", "dan", "mi",
    "mı", "mu", "mü", "ne", "o", "ya", "ki", "ama", "en", "çok", "daha",
    "olan", "oldu", "olarak", "olan", "var", "yok", "gibi", "kadar",
    "sonra", "önce", "her", "biz", "siz", "onlar", "ben", "sen",
    "bu", "şu", "hangi", "nasıl", "neden", "çünkü", "ancak", "fakat",
    "hem", "veya", "ya", "yani", "ise", "iken", "diye", "göre",
    "üzere", "karşı", "doğru", "içinde", "üzerinde", "altında",
]

mlflow.set_tracking_uri(f"sqlite:///{_REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("news-clustering")

# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def _build_corpus(items: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    """Return (indices, texts) for Turkish items with enough content."""
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


def _cluster(
    texts: list[str],
    n_clusters: int,
) -> tuple[np.ndarray, TfidfVectorizer, np.ndarray]:
    """Fit TF-IDF + KMeans and return (labels, vectorizer, tfidf_matrix)."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=_TURKISH_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    n_clusters = min(n_clusters, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)

    return labels, vectorizer, matrix


def _top_keywords(
    cluster_id: int,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int,
) -> list[str]:
    """Return top TF-IDF terms for *cluster_id*."""
    mask = labels == cluster_id
    if not mask.any():
        return []
    centroid = np.asarray(matrix[mask].mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    top_indices = centroid.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices]


def _build_cluster_summaries(
    items: list[dict[str, Any]],
    indices: list[int],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    n_clusters: int,
) -> list[dict[str, Any]]:
    """Build a summary record for each cluster."""
    summaries = []
    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for pos, item_idx in enumerate(indices):
        cluster_indices[int(labels[pos])].append(item_idx)

    for cid in range(n_clusters):
        item_idxs = cluster_indices.get(cid, [])
        keywords = _top_keywords(cid, labels, vectorizer, matrix, TOP_KEYWORDS_PER_CLUSTER)
        sources = Counter(items[i]["source_name"] for i in item_idxs)
        sentiments = Counter(
            items[i].get("sentiment_label") for i in item_idxs
            if items[i].get("sentiment_label")
        )
        summaries.append({
            "cluster_id": cid,
            "size": len(item_idxs),
            "keywords": keywords,
            "sources": dict(sources.most_common()),
            "sentiment_distribution": dict(sentiments),
        })

    summaries.sort(key=lambda x: x["size"], reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_analyzed(date_str: str, analyzed_dir: Path) -> list[dict[str, Any]]:
    path = analyzed_dir / f"{date_str}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Analyzed file not found: {path}\n"
            f"Run `python -m src.analysis.ner --date {date_str}` first."
        )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info(f"Loaded {len(data)} items from {path.name}")
    return data


def _save_analyzed(items: list[dict[str, Any]], date_str: str, analyzed_dir: Path) -> Path:
    path = analyzed_dir / f"{date_str}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(items)} clustered items → {path}")
    return path


def _save_cluster_summaries(
    summaries: list[dict[str, Any]],
    date_str: str,
    analyzed_dir: Path,
) -> Path:
    path = analyzed_dir / f"{date_str}_clusters.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(summaries)} cluster summaries → {path}")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_topics(
    date_str: str | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    analyzed_dir: Path | None = None,
) -> Path:
    """Run topic clustering for a single day's analyzed file.

    Args:
        date_str: ISO date string (``"YYYY-MM-DD"``).  Defaults to today.
        n_clusters: Number of topic clusters.  Defaults to
            ``DEFAULT_N_CLUSTERS``.
        analyzed_dir: Source/destination directory.  Defaults to
            ``data/analyzed/``.

    Returns:
        Path to the updated analyzed JSON file.
    """
    date_str = date_str or date.today().isoformat()
    analyzed_dir = analyzed_dir or _DATA_ANALYZED_DIR

    items = _load_analyzed(date_str, analyzed_dir)

    t0 = time.perf_counter()
    indices, texts = _build_corpus(items)

    if len(texts) < n_clusters:
        logger.warning(
            f"Only {len(texts)} usable items — reducing n_clusters to {len(texts)}"
        )
        n_clusters = max(2, len(texts))

    logger.info(f"Clustering {len(texts)} items into {n_clusters} topics...")
    labels, vectorizer, matrix = _cluster(texts, n_clusters)

    sil_score = float(silhouette_score(matrix, labels, metric="cosine"))
    duration = time.perf_counter() - t0
    logger.info(f"Clustering done — silhouette={sil_score:.4f} in {duration:.1f}s")

    # Attach cluster fields to items
    item_to_cluster: dict[int, int] = {}
    item_keywords: dict[int, list[str]] = {}
    for cid in range(n_clusters):
        kws = _top_keywords(cid, labels, vectorizer, matrix, TOP_KEYWORDS_PER_CLUSTER)
        for pos, item_idx in enumerate(indices):
            if labels[pos] == cid:
                item_to_cluster[item_idx] = cid
                item_keywords[item_idx] = kws

    enriched = []
    for i, item in enumerate(items):
        result = dict(item)
        result["cluster_id"] = item_to_cluster.get(i)
        result["cluster_keywords"] = item_keywords.get(i, [])
        enriched.append(result)

    summaries = _build_cluster_summaries(items, indices, labels, vectorizer, matrix, n_clusters)
    _save_cluster_summaries(summaries, date_str, analyzed_dir)
    out_path = _save_analyzed(enriched, date_str, analyzed_dir)

    with mlflow.start_run(run_name=f"clustering-{date_str}"):
        mlflow.log_params({
            "date": date_str,
            "n_clusters": n_clusters,
            "corpus_size": len(texts),
            "top_keywords_per_cluster": TOP_KEYWORDS_PER_CLUSTER,
        })
        mlflow.log_metrics({
            "silhouette_score": sil_score,
            "duration_sec": duration,
            "avg_cluster_size": len(texts) / n_clusters,
        })

    logger.info("Top clusters:")
    for s in summaries[:5]:
        logger.info(
            f"  Cluster {s['cluster_id']} ({s['size']} items): {', '.join(s['keywords'][:4])}"
        )

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Turkish news items by topic."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        metavar="YYYY-MM-DD",
        help="Date of the analyzed file to cluster (default: today)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        metavar="N",
        help=f"Number of topic clusters (default: {DEFAULT_N_CLUSTERS})",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    try:
        cluster_topics(date_str=args.date, n_clusters=args.n_clusters)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
