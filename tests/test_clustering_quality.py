from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analysis.clustering import (
    _build_cluster_summaries,
    _cluster_hdbscan,
    _cluster_quality_metrics,
    _generate_title,
    _quality_warnings,
    _top_keywords,
)


def test_german_articles_are_not_standalone_keywords_but_article_phrases_survive():
    texts = [
        "das sofa ist neu",
        "das sofa ist bequem",
        "das sofa steht im haus",
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None, min_df=1)
    matrix = vectorizer.fit_transform(texts)
    labels = np.array([0, 0, 0])

    keywords = _top_keywords(
        0,
        labels,
        vectorizer,
        matrix,
        top_n=5,
        stopwords={"das", "ist", "im"},
        language="de",
    )

    assert "das" not in keywords
    assert "das sofa" in keywords
    assert "sofa" not in keywords


def test_cluster_quality_flags_dominant_cluster_and_stopword_keywords():
    metrics = _cluster_quality_metrics(
        [
            {"size": 90, "keywords": ["die"]},
            {"size": 10, "keywords": ["wirtschaft"]},
        ],
        corpus_size=100,
        stopwords={"die"},
        unknown_category_pct=5.0,
    )
    metrics["silhouette_score"] = 0.05

    assert metrics["largest_cluster_pct"] == 90.0
    assert metrics["stopword_keyword_rate"] == 50.0
    assert set(_quality_warnings(metrics)) == {
        "low_silhouette",
        "dominant_cluster",
        "stopword_keywords",
    }


def test_quality_warnings_flags_high_noise():
    assert "high_noise" in _quality_warnings({"noise_pct": 55.0})
    assert "high_noise" not in _quality_warnings({"noise_pct": 10.0})


def test_hdbscan_finds_dense_clusters_and_marks_outliers_as_noise():
    rng = np.random.default_rng(0)
    blob1 = np.array([10.0, 0.0]) + rng.normal(0, 0.3, (8, 2))
    blob2 = np.array([0.0, 10.0]) + rng.normal(0, 0.3, (8, 2))
    outlier = np.array([[-12.0, -12.0]])
    embeddings = np.vstack([blob1, blob2, outlier])

    labels = _cluster_hdbscan(embeddings, min_cluster_size=5)

    clusters = sorted({int(label) for label in labels if label >= 0})
    assert len(clusters) == 2
    # The far-away point fits no dense topic → noise.
    assert int(labels[-1]) == -1


def test_build_cluster_summaries_only_covers_passed_cluster_ids():
    items = [
        {"source_name": "A", "sentiment_label": "positive"},
        {"source_name": "B", "sentiment_label": "negative"},
        {"source_name": "A", "sentiment_label": None},
    ]
    indices = [0, 1, 2]
    labels = np.array([0, 0, -1])  # item 2 is HDBSCAN noise
    texts = ["ekonomi enflasyon zam", "ekonomi enflasyon faiz", "tamamen baska konu"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None, min_df=1)
    matrix = vectorizer.fit_transform(texts)

    summaries = _build_cluster_summaries(
        items,
        indices,
        labels,
        vectorizer,
        matrix,
        cluster_ids=[0],  # noise (-1) deliberately excluded
        stopwords=set(),
        language="tr",
    )

    assert [s["cluster_id"] for s in summaries] == [0]
    assert summaries[0]["size"] == 2  # noise item not counted


def test_generate_title_skips_wordpiece_entity_fragments():
    # Legacy TR rows carry "##mgrup"-style fragments; the title must never
    # surface them ("Yapay · ##mgrup"). With only fragments, fall back to
    # the keyword alone.
    items = [
        {"entities": {"ORG": ["##mgrup"], "LOC": [], "PER": []}},
        {"entities": {"ORG": ["##mgrup"], "LOC": [], "PER": []}},
    ]
    title = _generate_title(["yapay"], items)
    assert "##" not in title
    assert title == "Yapay"


def test_generate_title_uses_clean_entity_over_fragment():
    items = [
        {"entities": {"PER": ["Erdoğan", "##gan"], "ORG": [], "LOC": []}},
        {"entities": {"PER": ["Erdoğan"], "ORG": [], "LOC": []}},
    ]
    title = _generate_title(["siyaset"], items)
    assert title == "Siyaset · Erdoğan"
