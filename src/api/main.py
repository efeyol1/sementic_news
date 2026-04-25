"""FastAPI application for Semantic News TR."""

from collections import Counter, defaultdict
from datetime import date

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.analysis.vector_store import find_similar
from src.db.queries import (
    fetch_all_for_api,
    fetch_available_dates,
    fetch_cluster_summaries,
)
from src.db.schema import init_db

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic News TR API",
    description="""
Türk haber kaynaklarından günlük olarak toplanan haberlerin semantik analiz sonuçlarını sunar.

## Pipeline
Her gün sabah 07:00'de çalışır:
1. **RSS Collect** — 10 kaynaktan haber çeker
2. **Preprocess** — HTML temizleme, dil tespiti
3. **Sentiment** — BERT tabanlı pozitif/negatif sınıflandırma
4. **NER** — Kişi, kurum, yer adı çıkarma
5. **Clustering** — TF-IDF + KMeans ile 15 konu kümesi
6. **Vector Store** — ChromaDB semantik arama indeksi

## Kaynaklar
Habertürk · Hürriyet · NTV · CNN Türk · Sözcü · Milliyet · Sabah · TRT Haber · Cumhuriyet · Yeni Şafak
""",
    version="2.0.0",
    contact={"name": "Semantic News TR", "url": "https://github.com/efeyol11/sementic_news"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Analiz", "description": "Günlük haber analizi endpoint'leri"},
        {"name": "Sistem", "description": "Sağlık kontrolü ve meta bilgiler"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def _startup():
    try:
        init_db()
    except Exception as exc:
        import logging
        logging.warning(f"DB init on startup failed: {exc}")


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class SentimentCounts(BaseModel):
    positive: int = Field(..., example=264)
    negative: int = Field(..., example=276)


class SentimentPercentages(BaseModel):
    positive: float = Field(..., example=48.9)
    negative: float = Field(..., example=51.1)


class SentimentSummary(BaseModel):
    counts: SentimentCounts
    percentages: SentimentPercentages


class ClusterSummary(BaseModel):
    cluster_id: int = Field(..., example=3)
    title: str = Field(..., example="Ekonomi · Merkez Bankası")
    size: int = Field(..., example=47)
    keywords: list[str] = Field(..., example=["ekonomi", "dolar", "faiz"])


class TopEntities(BaseModel):
    PER: list[str] = Field(..., example=["Erdoğan", "Trump"])
    ORG: list[str] = Field(..., example=["TBMM", "Merkez Bankası"])
    LOC: list[str] = Field(..., example=["Ankara", "İstanbul"])


class TodayResponse(BaseModel):
    date: str = Field(..., example="2026-04-20")
    total_items: int = Field(..., example=545)
    turkish_items: int = Field(..., example=540)
    sources: dict[str, int] = Field(..., example={"Cumhuriyet": 114})
    sentiment: SentimentSummary
    top_entities: TopEntities
    cluster_count: int = Field(..., example=15)
    top_clusters: list[ClusterSummary]
    available_dates: list[str] = Field(..., example=["2026-04-20"])


class NewsItem(BaseModel):
    title: str = Field(..., example="Kabine toplantısı ne zaman?")
    source_name: str = Field(..., example="Habertürk")
    published_date: str = Field(..., example="2026-04-20T07:30:00+00:00")
    sentiment_label: str | None = Field(None, example="positive")
    sentiment_score: float | None = Field(None, example=0.977)
    entities: dict[str, list[str]] | None = Field(None)
    link: str | None = Field(None)


class TopicResponse(BaseModel):
    date: str = Field(..., example="2026-04-20")
    cluster_id: int = Field(..., example=3)
    keywords: list[str]
    size: int = Field(..., example=47)
    sentiment_distribution: dict[str, int]
    news: list[NewsItem]


class SourceStats(BaseModel):
    total: int = Field(..., example=109)
    sentiment_counts: dict[str, int]
    sentiment_percentages: dict[str, float]
    avg_confidence: float | None = Field(None, example=0.861)


class SourceComparisonResponse(BaseModel):
    date: str = Field(..., example="2026-04-20")
    sources: dict[str, SourceStats]


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class SimilarNewsItem(BaseModel):
    title: str
    source_name: str
    date: str
    sentiment_label: str
    link: str | None
    similarity: float = Field(..., example=0.91)


class SimilarNewsResponse(BaseModel):
    query_title: str
    results: list[SimilarNewsItem]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _get_items(date_str: str) -> list[dict]:
    items = fetch_all_for_api(date_str)
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} tarihine ait analiz verisi bulunamadı.",
        )
    return items


def _get_clusters(date_str: str) -> list[dict]:
    clusters = fetch_cluster_summaries(date_str)
    if not clusters:
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} tarihine ait cluster verisi bulunamadı.",
        )
    return clusters


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Sistem"], response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get(
    "/api/today",
    tags=["Analiz"],
    summary="Günlük analiz özeti",
    response_model=TodayResponse,
    responses={404: {"description": "Belirtilen tarihe ait veri bulunamadı"}},
)
def today(
    date_str: str = Query(
        default=None,
        alias="date",
        description="Analiz tarihi (YYYY-MM-DD). Belirtilmezse bugünün verisi döner.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    target = date_str or date.today().isoformat()
    items = _get_items(target)
    clusters = fetch_cluster_summaries(target)

    turkish = [i for i in items if i.get("is_turkish")]
    n = len(turkish)

    _counts = Counter(i.get("sentiment_label") for i in turkish if i.get("sentiment_label"))
    sentiment_counts = {"positive": _counts.get("positive", 0), "negative": _counts.get("negative", 0)}
    sentiment_pct = (
        {k: round(v / n * 100, 1) for k, v in sentiment_counts.items()}
        if n else {"positive": 0.0, "negative": 0.0}
    )

    entity_agg: dict[str, Counter] = {"PER": Counter(), "ORG": Counter(), "LOC": Counter()}
    for item in turkish:
        for label, words in (item.get("entities") or {}).items():
            if label in entity_agg:
                entity_agg[label].update(words)
    top_entities = {label: [w for w, _ in ctr.most_common(10)] for label, ctr in entity_agg.items()}

    source_counts = Counter(i["source_name"] for i in items)

    return {
        "date": target,
        "total_items": len(items),
        "turkish_items": n,
        "sources": dict(source_counts.most_common()),
        "sentiment": {"counts": dict(sentiment_counts), "percentages": sentiment_pct},
        "top_entities": top_entities,
        "cluster_count": len(clusters),
        "top_clusters": [
            {
                "cluster_id": c["cluster_id"],
                "title": c.get("title", c["keywords"][0] if c["keywords"] else ""),
                "size": c["size"],
                "keywords": c["keywords"][:5],
            }
            for c in sorted(clusters, key=lambda x: x["size"], reverse=True)[:5]
        ],
        "available_dates": fetch_available_dates(),
    }


@app.get(
    "/api/topic/{cluster_id}",
    tags=["Analiz"],
    summary="Konu kümesi detayı",
    response_model=TopicResponse,
    responses={404: {"description": "Belirtilen cluster_id veya tarihe ait veri bulunamadı"}},
)
def topic(
    cluster_id: int = Path(..., description="Küme numarası (0-14)", ge=0, le=14),
    date_str: str = Query(
        default=None,
        alias="date",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    target = date_str or date.today().isoformat()
    items = _get_items(target)
    clusters = fetch_cluster_summaries(target)

    cluster_meta = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
    if cluster_meta is None:
        raise HTTPException(status_code=404, detail=f"cluster_id={cluster_id} bulunamadı.")

    news = [
        {
            "title": i["title"],
            "source_name": i["source_name"],
            "published_date": str(i.get("published_date", "")),
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


@app.get(
    "/api/source-comparison",
    tags=["Analiz"],
    summary="Kaynak bazlı sentiment karşılaştırması",
    response_model=SourceComparisonResponse,
    responses={404: {"description": "Belirtilen tarihe ait veri bulunamadı"}},
)
def source_comparison(
    date_str: str = Query(
        default=None,
        alias="date",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    target = date_str or date.today().isoformat()
    items = _get_items(target)

    sources: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "sentiment_counts": Counter(), "scores": [],
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
                if data["scores"] else None
            ),
        }

    return {"date": target, "sources": result}


@app.get(
    "/api/similar",
    tags=["Analiz"],
    summary="Semantik olarak benzer haberler",
    response_model=SimilarNewsResponse,
    responses={404: {"description": "Vektör veritabanı boş veya haber bulunamadı"}},
)
def similar_news(
    q: str = Query(..., description="Aranacak haber başlığı veya metin", min_length=5),
    n: int = Query(default=5, ge=1, le=20),
):
    results = find_similar(q, n=n)
    if not results:
        raise HTTPException(status_code=404, detail="Henüz hiç embedding yok — pipeline çalıştırın.")
    return {
        "query_title": q,
        "results": [
            {
                "title": r.get("title", ""),
                "source_name": r.get("source_name", ""),
                "date": r.get("date", ""),
                "sentiment_label": r.get("sentiment_label", ""),
                "link": r.get("link") or None,
                "similarity": r["similarity"],
            }
            for r in results
        ],
    }
