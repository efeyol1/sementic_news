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

from src.analysis.text_inputs import build_clustering_text
from src.config import load_country_config
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

_GERMAN_ARTICLES = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "eines", "einem", "einen",
}

_DEFAULT_CLUSTER_QUALITY_THRESHOLDS = {
    "silhouette_min": 0.10,
    "largest_cluster_pct_max": 25.0,
    "stopword_keyword_rate_max": 10.0,
    # HDBSCAN-only: fraction of items left unclustered (noise). Some noise is
    # healthy density separation; a very high rate means the params are too
    # strict (or the embedding is too tangled) and topics are being lost.
    "noise_pct_max": 40.0,
}

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
        text = build_clustering_text(item)
        if len(text) >= 20:
            indices.append(i)
            texts.append(text)
    return indices, texts


def _embed(texts: list[str], model_id: str = _EMBED_MODEL) -> np.ndarray:
    logger.info(f"Encoding {len(texts)} texts with {model_id}...")
    model = SentenceTransformer(model_id)
    return model.encode(texts, batch_size=64, show_progress_bar=True)


def _cluster_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    n_clusters = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def _cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Density-based clustering. Items that fit no dense topic get label -1.

    Embeddings are L2-normalized so that Euclidean distance is monotonic with
    cosine distance — the same metric KMeans/silhouette use here — without
    HDBSCAN having to support a cosine metric directly.
    """
    import hdbscan
    from sklearn.preprocessing import normalize

    normed = normalize(np.asarray(embeddings, dtype=np.float64))
    if len(normed) < 3:
        # HDBSCAN needs at least a few points; mark everything as noise.
        return np.full(len(normed), -1, dtype=int)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=float(cluster_selection_epsilon),
    )
    return clusterer.fit_predict(normed)


def _reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
) -> np.ndarray:
    """UMAP dimensionality reduction before HDBSCAN.

    HDBSCAN's density estimate is unreliable on raw 384-dim embeddings (most
    items become noise). Reducing to ~10 dims with a cosine metric first gives
    dense, well-separated neighborhoods — the same pre-step BERTopic uses.
    """
    import umap

    data = np.asarray(embeddings, dtype=np.float64)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(2, len(data) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(data)


def _top_keywords(
    cluster_id: int,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int,
    stopwords: set[str] | None = None,
    language: str = "tr",
) -> list[str]:
    mask = labels == cluster_id
    if not mask.any():
        return []
    centroid = np.asarray(matrix[mask].mean(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    keywords: list[str] = []
    article_phrases: dict[str, str] = {}
    if language == "de":
        for idx in np.flatnonzero(centroid):
            normalized = _normalize_keyword(str(feature_names[idx]))
            tokens = normalized.split()
            if (
                len(tokens) == 2
                and tokens[0] in _GERMAN_ARTICLES
                and tokens[1] not in (stopwords or set())
            ):
                article_phrases.setdefault(tokens[1], normalized)
    for idx in centroid.argsort()[::-1]:
        feature = str(feature_names[idx])
        normalized = _normalize_keyword(feature)
        if not _is_informative_keyword(normalized, stopwords or set()):
            continue
        if language == "de":
            tokens = normalized.split()
            if len(tokens) == 1 and tokens[0] in article_phrases:
                normalized = article_phrases[tokens[0]]
        if normalized not in keywords:
            keywords.append(normalized)
        if len(keywords) >= top_n:
            break
    return keywords


def _normalize_keyword(keyword: str) -> str:
    return " ".join(keyword.lower().split())


def _is_informative_keyword(keyword: str, stopwords: set[str]) -> bool:
    tokens = keyword.split()
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] not in stopwords and len(tokens[0]) > 1
    return any(token not in stopwords for token in tokens)


def _cluster_quality_metrics(
    summaries: list[dict[str, Any]],
    corpus_size: int,
    stopwords: set[str],
    unknown_category_pct: float | None = None,
) -> dict[str, float]:
    if not summaries or not corpus_size:
        return {
            "largest_cluster_pct": 0.0,
            "small_cluster_pct": 0.0,
            "stopword_keyword_rate": 0.0,
            "unknown_category_pct": unknown_category_pct or 0.0,
        }

    largest = max(summary.get("size", 0) for summary in summaries)
    small = sum(1 for summary in summaries if summary.get("size", 0) < 5)
    keywords = [
        keyword
        for summary in summaries
        for keyword in summary.get("keywords", [])
    ]
    stopword_keywords = sum(
        1 for keyword in keywords
        if not _is_informative_keyword(_normalize_keyword(str(keyword)), stopwords)
    )
    return {
        "largest_cluster_pct": round(largest / corpus_size * 100, 2),
        "small_cluster_pct": round(small / len(summaries) * 100, 2),
        "stopword_keyword_rate": round(
            stopword_keywords / len(keywords) * 100, 2
        ) if keywords else 0.0,
        "unknown_category_pct": round(unknown_category_pct or 0.0, 2),
    }


def _quality_warnings(metrics: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if metrics.get("silhouette_score", 0.0) < _DEFAULT_CLUSTER_QUALITY_THRESHOLDS["silhouette_min"]:
        warnings.append("low_silhouette")
    if metrics.get("largest_cluster_pct", 0.0) > _DEFAULT_CLUSTER_QUALITY_THRESHOLDS["largest_cluster_pct_max"]:
        warnings.append("dominant_cluster")
    if metrics.get("stopword_keyword_rate", 0.0) > _DEFAULT_CLUSTER_QUALITY_THRESHOLDS["stopword_keyword_rate_max"]:
        warnings.append("stopword_keywords")
    if metrics.get("noise_pct", 0.0) > _DEFAULT_CLUSTER_QUALITY_THRESHOLDS["noise_pct_max"]:
        warnings.append("high_noise")
    return warnings


def _unknown_category_pct(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    unknown = sum(
        1 for item in items
        if (item.get("canonical_category") or item.get("category") or "unknown") == "unknown"
    )
    return unknown / len(items) * 100


def _clustering_config(country_code: str, country_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = country_config or load_country_config(country_code)
    return config.get("clustering") or {}


def _generate_title(keywords: list[str], cluster_items: list[dict]) -> str:
    entity_counter: Counter = Counter()
    for item in cluster_items:
        for label in ("ORG", "LOC", "PER"):
            ents = (item.get("entities") or {}).get(label, [])
            # Legacy TR items stored WordPiece fragments ("##mgrup", "##usya")
            # in their entities; skip them so titles don't read
            # "Yapay · ##mgrup". Fresh NER runs no longer emit these (filtered
            # at the source in ner._extract_entities), but old rows persist
            # until re-processed.
            entity_counter.update(e for e in ents if e and "##" not in e)

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
    cluster_ids: list[int],
    stopwords: set[str],
    language: str,
) -> list[dict[str, Any]]:
    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for pos, item_idx in enumerate(indices):
        cluster_indices[int(labels[pos])].append(item_idx)

    summaries = []
    for cid in cluster_ids:
        item_idxs = cluster_indices.get(cid, [])
        cluster_items = [items[i] for i in item_idxs]
        keywords = _top_keywords(
            cid,
            labels,
            vectorizer,
            matrix,
            TOP_KEYWORDS_PER_CLUSTER,
            stopwords=stopwords,
            language=language,
        )
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
    country_code: str = "TR",
    country_config: dict[str, Any] | None = None,
) -> int:
    """Run topic clustering for a single day's items.

    Returns:
        Number of items clustered.
    """
    date_str = date_str or date.today().isoformat()
    country_config = country_config or load_country_config(country_code)
    clustering_cfg = _clustering_config(country_code, country_config=country_config)
    language = country_config.get("language", "tr")
    method = str(clustering_cfg.get("method") or "kmeans").lower()
    n_clusters = int(n_clusters or clustering_cfg.get("default_n_clusters") or DEFAULT_N_CLUSTERS)
    embed_model = clustering_cfg.get("embedding_model") or _EMBED_MODEL
    stopwords = {
        _normalize_keyword(str(word))
        for word in (clustering_cfg.get("stopwords") or _TURKISH_STOPWORDS)
        if str(word).strip()
    }

    items = fetch_for_clustering(date_str, country_code=country_code)
    if not items:
        logger.warning(f"No items found for clustering on {date_str}")
        return 0

    t0 = time.perf_counter()
    indices, texts = _build_corpus(items)
    if len(texts) < 2:
        logger.warning(f"Only {len(texts)} usable items — skipping clustering")
        return 0

    if method == "hdbscan":
        embeddings = _embed(texts, model_id=embed_model)
        # Cluster in a reduced space when UMAP is enabled, but keep `embeddings`
        # (the full semantic space) for silhouette so the metric stays
        # comparable to the KMeans path.
        umap_cfg = clustering_cfg.get("umap") or {}
        cluster_space = embeddings
        if umap_cfg.get("enabled"):
            cluster_space = _reduce_umap(
                embeddings,
                n_components=int(umap_cfg.get("n_components") or 10),
                n_neighbors=int(umap_cfg.get("n_neighbors") or 15),
                min_dist=float(umap_cfg.get("min_dist") or 0.0),
            )
        labels = _cluster_hdbscan(
            cluster_space,
            min_cluster_size=int(clustering_cfg.get("min_cluster_size") or 5),
            min_samples=clustering_cfg.get("min_samples"),
            cluster_selection_epsilon=float(clustering_cfg.get("cluster_selection_epsilon") or 0.0),
        )
    else:
        if len(texts) < n_clusters:
            logger.warning(f"Only {len(texts)} usable items — reducing n_clusters to {len(texts)}")
            n_clusters = max(2, len(texts))
        embeddings = _embed(texts, model_id=embed_model)
        labels = _cluster_kmeans(embeddings, n_clusters)

    # HDBSCAN assigns -1 to "noise" — items that fit no dense topic. They are
    # excluded from cluster summaries and silhouette; KMeans never emits -1.
    cluster_ids = sorted({int(label) for label in labels if label >= 0})
    noise_mask = labels < 0
    noise_pct = round(float(noise_mask.sum()) / len(labels) * 100, 2) if len(labels) else 0.0

    non_noise = ~noise_mask
    if len(cluster_ids) >= 2 and int(non_noise.sum()) >= 2:
        sil_score = float(
            silhouette_score(embeddings[non_noise], labels[non_noise], metric="cosine")
        )
    else:
        sil_score = 0.0
    duration = time.perf_counter() - t0
    logger.info(
        f"Clustering done — method={method} clusters={len(cluster_ids)} "
        f"noise={noise_pct}% silhouette={sil_score:.4f} in {duration:.1f}s"
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    summaries = _build_cluster_summaries(
        items,
        indices,
        labels,
        vectorizer,
        matrix,
        cluster_ids,
        stopwords=stopwords,
        language=language,
    )
    title_map = {s["cluster_id"]: s["title"] for s in summaries}
    keywords_map = {s["cluster_id"]: s["keywords"] for s in summaries}
    unknown_category_pct = _unknown_category_pct([items[i] for i in indices])
    quality_metrics = _cluster_quality_metrics(
        summaries,
        corpus_size=len(texts),
        stopwords=stopwords,
        unknown_category_pct=unknown_category_pct,
    )
    quality_metrics["silhouette_score"] = round(sil_score, 4)
    quality_metrics["noise_pct"] = noise_pct
    quality_metrics["n_clusters"] = len(cluster_ids)
    warnings = _quality_warnings(quality_metrics)
    if warnings:
        logger.warning(f"Cluster quality warnings: {warnings} metrics={quality_metrics}")
    else:
        logger.info(f"Cluster quality metrics: {quality_metrics}")

    # Build per-item cluster update list. Reuses the keywords already computed
    # for the summaries. Noise items (label -1) are written back with
    # cluster_id=-1 / no keywords so a re-run clears any stale assignment.
    cluster_updates: list[dict[str, Any]] = []
    for pos, item_idx in enumerate(indices):
        cid = int(labels[pos])
        cluster_updates.append({
            "id": items[item_idx]["id"],
            "cluster_id": cid,
            "cluster_keywords": keywords_map.get(cid, []),
            "cluster_title": title_map.get(cid, ""),
        })

    bulk_update_clustering(cluster_updates)
    upsert_cluster_summaries(summaries, date_str, country_code=country_code)

    n_effective = len(cluster_ids)
    clustered_count = int(non_noise.sum())
    with mlflow.start_run(run_name=f"clustering-{date_str}"):
        mlflow.log_params({
            "date": date_str,
            "country_code": country_code,
            "method": method,
            "n_clusters": n_effective,
            "corpus_size": len(texts),
            "embed_model": embed_model,
            "language": language,
        })
        mlflow.log_metrics({
            "silhouette_score": sil_score,
            "duration_sec": duration,
            "avg_cluster_size": clustered_count / n_effective if n_effective else 0.0,
            **quality_metrics,
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
    parser.add_argument("--country", default="turkey", help="Country slug or ISO code (default: turkey)")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS, metavar="N")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_country_config(args.country)
    cluster_topics(
        date_str=args.date,
        n_clusters=args.n_clusters,
        country_code=cfg["country_code"],
        country_config=cfg,
    )
    sys.exit(0)
