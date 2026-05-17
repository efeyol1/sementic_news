from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analysis.clustering import (
    _cluster_quality_metrics,
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
