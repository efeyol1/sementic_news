"""FastAPI application for Semantic News TR.

Endpoints:
    GET /api/today                  — bugünün analiz özeti
    GET /api/topic/{cluster_id}     — belirli bir cluster'ın haberleri
    GET /api/source-comparison      — kaynak bazlı sentiment karşılaştırması
    GET /health                     — liveness probe
"""

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic News TR",
    description="Türk haber kaynaklarından günlük sentiment & NER & kümeleme analizi",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)  # /metrics endpoint'i açar

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYZED_DIR = _REPO_ROOT / "data" / "analyzed"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_analyzed(date_str: str) -> list[dict[str, Any]]:
    path = _ANALYZED_DIR / f"{date_str}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} tarihine ait analiz verisi bulunamadı.",
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_clusters(date_str: str) -> list[dict[str, Any]]:
    path = _ANALYZED_DIR / f"{date_str}_clusters.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} tarihine ait cluster verisi bulunamadı.",
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _available_dates() -> list[str]:
    return sorted(
        p.stem for p in _ANALYZED_DIR.glob("????-??-??.json")
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/today")
def today(
    date_str: str = Query(
        default=None, alias="date", description="YYYY-MM-DD, default: bugün"
    ),
):
    """Bir günün analiz özetini döndürür.

    - Sentiment dağılımı (positive / neutral / negative yüzdeleri)
    - En sık geçen entity'ler (PER, ORG, LOC)
    - Cluster sayısı ve en büyük 5 cluster özeti
    - Haber sayısı ve kaynak dağılımı
    """
    target = date_str or date.today().isoformat()
    items = _load_analyzed(target)
    clusters = _load_clusters(target)

    turkish = [i for i in items if i.get("is_turkish")]
    n = len(turkish)

    # Sentiment dağılımı
    sentiment_counts = Counter(
        i.get("sentiment_label") for i in turkish if i.get("sentiment_label")
    )
    sentiment_pct = (
        {k: round(v / n * 100, 1) for k, v in sentiment_counts.items()} if n else {}
    )

    # Top entity'ler
    entity_agg: dict[str, Counter] = {
        "PER": Counter(), "ORG": Counter(), "LOC": Counter()
    }
    for item in turkish:
        for label, words in (item.get("entities") or {}).items():
            if label in entity_agg:
                entity_agg[label].update(words)
    top_entities = {
        label: [w for w, _ in ctr.most_common(10)]
        for label, ctr in entity_agg.items()
    }

    # Kaynak dağılımı
    source_counts = Counter(i["source_name"] for i in items)

    return {
        "date": target,
        "total_items": len(items),
        "turkish_items": n,
        "sources": dict(source_counts.most_common()),
        "sentiment": {
            "counts": dict(sentiment_counts),
            "percentages": sentiment_pct,
        },
        "top_entities": top_entities,
        "cluster_count": len(clusters),
        "top_clusters": [
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "keywords": c["keywords"][:5],
            }
            for c in sorted(clusters, key=lambda x: x["size"], reverse=True)[:5]
        ],
        "available_dates": _available_dates(),
    }


@app.get("/api/topic/{cluster_id}")
def topic(cluster_id: int, date_str: str = Query(default=None, alias="date")):
    """Belirli bir cluster'daki haberleri döndürür.

    Her haber için: title, source_name, published_date, sentiment_label,
    sentiment_score, entities, link alanları döndürülür.
    """
    target = date_str or date.today().isoformat()
    items = _load_analyzed(target)
    clusters = _load_clusters(target)

    cluster_meta = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
    if cluster_meta is None:
        raise HTTPException(
            status_code=404, detail=f"cluster_id={cluster_id} bulunamadı."
        )

    news = [
        {
            "title": i["title"],
            "source_name": i["source_name"],
            "published_date": i["published_date"],
            "sentiment_label": i.get("sentiment_label"),
            "sentiment_score": i.get("sentiment_score"),
            "entities": i.get("entities"),
            "link": i.get("link"),
        }
        for i in items
        if i.get("cluster_id") == cluster_id
    ]

    return {
        "date": target,
        "cluster_id": cluster_id,
        "keywords": cluster_meta["keywords"],
        "size": cluster_meta["size"],
        "sentiment_distribution": cluster_meta["sentiment_distribution"],
        "news": news,
    }


@app.get("/api/source-comparison")
def source_comparison(date_str: str = Query(default=None, alias="date")):
    """Kaynak bazlı sentiment karşılaştırmasını döndürür.

    Her kaynak için: haber sayısı, sentiment dağılımı (%), ortalama güven skoru.
    """
    target = date_str or date.today().isoformat()
    items = _load_analyzed(target)

    sources: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "sentiment_counts": Counter(),
        "scores": [],
    })

    for item in items:
        if not item.get("is_turkish"):
            continue
        src = item["source_name"]
        sources[src]["total"] += 1
        if item.get("sentiment_label"):
            sources[src]["sentiment_counts"][item["sentiment_label"]] += 1
        if item.get("sentiment_score") is not None:
            sources[src]["scores"].append(item["sentiment_score"])

    result = {}
    for src, data in sorted(sources.items()):
        n = data["total"]
        counts = dict(data["sentiment_counts"])
        result[src] = {
            "total": n,
            "sentiment_counts": counts,
            "sentiment_percentages": (
                {k: round(v / n * 100, 1) for k, v in counts.items()} if n else {}
            ),
            "avg_confidence": (
                round(sum(data["scores"]) / len(data["scores"]), 4)
                if data["scores"]
                else None
            ),
        }

    return {"date": target, "sources": result}
