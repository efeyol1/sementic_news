"""FastAPI application for Semantic News TR."""

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path as FilePath
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

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

## Kaynaklar
Habertürk · Hürriyet · NTV · CNN Türk · Sözcü · Milliyet · Sabah · TRT Haber · Cumhuriyet · Yeni Şafak
""",
    version="1.0.0",
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

_REPO_ROOT = FilePath(__file__).resolve().parents[2]
_ANALYZED_DIR = _REPO_ROOT / "data" / "analyzed"

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class SentimentCounts(BaseModel):
    positive: int = Field(..., description="Pozitif haber sayısı", example=264)
    negative: int = Field(..., description="Negatif haber sayısı", example=276)


class SentimentPercentages(BaseModel):
    positive: float = Field(..., description="Pozitif yüzde (0-100)", example=48.9)
    negative: float = Field(..., description="Negatif yüzde (0-100)", example=51.1)


class SentimentSummary(BaseModel):
    counts: SentimentCounts
    percentages: SentimentPercentages


class ClusterSummary(BaseModel):
    cluster_id: int = Field(..., example=3)
    title: str = Field(..., description="Küme başlığı", example="Ekonomi · Merkez Bankası")
    size: int = Field(..., description="Kümedeki haber sayısı", example=47)
    keywords: list[str] = Field(..., description="En ayırt edici 5 kelime", example=["ekonomi", "dolar", "faiz"])


class TopEntities(BaseModel):
    PER: list[str] = Field(..., description="En çok geçen 10 kişi adı", example=["Erdoğan", "Trump"])
    ORG: list[str] = Field(..., description="En çok geçen 10 kurum adı", example=["TBMM", "Merkez Bankası"])
    LOC: list[str] = Field(..., description="En çok geçen 10 yer adı", example=["Ankara", "İstanbul"])


class TodayResponse(BaseModel):
    date: str = Field(..., description="Analiz tarihi (YYYY-MM-DD)", example="2026-04-20")
    total_items: int = Field(..., description="Toplam çekilen haber sayısı", example=545)
    turkish_items: int = Field(..., description="Türkçe olarak tespit edilen haber sayısı", example=540)
    sources: dict[str, int] = Field(..., description="Kaynak → haber sayısı", example={"Cumhuriyet": 114})
    sentiment: SentimentSummary
    top_entities: TopEntities
    cluster_count: int = Field(..., description="Toplam küme sayısı", example=15)
    top_clusters: list[ClusterSummary] = Field(..., description="En büyük 5 küme")
    available_dates: list[str] = Field(..., description="Veri mevcut tarihler", example=["2026-04-20"])


class NewsItem(BaseModel):
    title: str = Field(..., description="Haber başlığı", example="Kabine toplantısı ne zaman?")
    source_name: str = Field(..., description="Kaynak adı", example="Habertürk")
    published_date: str = Field(..., description="Yayın tarihi (ISO-8601)", example="2026-04-20T07:30:00+00:00")
    sentiment_label: str | None = Field(None, description="positive veya negative", example="positive")
    sentiment_score: float | None = Field(None, description="Model güven skoru (0-1)", example=0.977)
    entities: dict[str, list[str]] | None = Field(
        None, description="NER sonuçları", example={"PER": ["Erdoğan"], "ORG": ["TBMM"], "LOC": ["Ankara"]}
    )
    link: str | None = Field(None, description="Haberin orijinal URL'i")


class TopicResponse(BaseModel):
    date: str = Field(..., example="2026-04-20")
    cluster_id: int = Field(..., example=3)
    keywords: list[str] = Field(..., description="Kümenin tüm anahtar kelimeleri")
    size: int = Field(..., description="Kümede kaç haber var", example=47)
    sentiment_distribution: dict[str, int] = Field(
        ..., description="Küme içi sentiment sayıları", example={"positive": 25, "negative": 22}
    )
    news: list[NewsItem] = Field(..., description="Kümedeki tüm haberler")


class SourceStats(BaseModel):
    total: int = Field(..., description="Bu kaynaktan gelen haber sayısı", example=109)
    sentiment_counts: dict[str, int] = Field(..., example={"positive": 43, "negative": 66})
    sentiment_percentages: dict[str, float] = Field(
        ..., description="Yüzde cinsinden dağılım (0-100)", example={"positive": 39.4, "negative": 60.6}
    )
    avg_confidence: float | None = Field(None, description="Ortalama model güven skoru", example=0.861)


class SourceComparisonResponse(BaseModel):
    date: str = Field(..., example="2026-04-20")
    sources: dict[str, SourceStats] = Field(..., description="Kaynak adı → istatistikler")


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")

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
    return sorted(p.stem for p in _ANALYZED_DIR.glob("????-??-??.json"))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["Sistem"],
    summary="Servis sağlık kontrolü",
    response_model=HealthResponse,
)
def health():
    """API'nin ayakta olup olmadığını kontrol eder. Kubernetes/Docker liveness probe olarak kullanılır."""
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
    """
    Bir günün tam analiz özetini döndürür:

    - **Haber sayıları** — toplam ve Türkçe
    - **Kaynak dağılımı** — hangi kaynaktan kaç haber
    - **Sentiment** — pozitif/negatif sayı ve yüzdeleri
    - **Top entity'ler** — en çok geçen kişi, kurum ve yer adları
    - **Konu kümeleri** — en büyük 5 küme ve anahtar kelimeleri
    - **Mevcut tarihler** — sorgulanabilir tüm tarihler
    """
    target = date_str or date.today().isoformat()
    items = _load_analyzed(target)
    clusters = _load_clusters(target)

    turkish = [i for i in items if i.get("is_turkish")]
    n = len(turkish)

    sentiment_counts = Counter(
        i.get("sentiment_label") for i in turkish if i.get("sentiment_label")
    )
    sentiment_pct = (
        {k: round(v / n * 100, 1) for k, v in sentiment_counts.items()} if n else {}
    )

    entity_agg: dict[str, Counter] = {"PER": Counter(), "ORG": Counter(), "LOC": Counter()}
    for item in turkish:
        for label, words in (item.get("entities") or {}).items():
            if label in entity_agg:
                entity_agg[label].update(words)
    top_entities = {
        label: [w for w, _ in ctr.most_common(10)]
        for label, ctr in entity_agg.items()
    }

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
            {"cluster_id": c["cluster_id"], "title": c.get("title", c["keywords"][0] if c["keywords"] else ""), "size": c["size"], "keywords": c["keywords"][:5]}
            for c in sorted(clusters, key=lambda x: x["size"], reverse=True)[:5]
        ],
        "available_dates": _available_dates(),
    }


@app.get(
    "/api/topic/{cluster_id}",
    tags=["Analiz"],
    summary="Konu kümesi detayı",
    response_model=TopicResponse,
    responses={
        404: {"description": "Belirtilen cluster_id veya tarihe ait veri bulunamadı"},
    },
)
def topic(
    cluster_id: int = Path(..., description="Küme numarası (0-14)", ge=0, le=14),
    date_str: str = Query(
        default=None,
        alias="date",
        description="Analiz tarihi (YYYY-MM-DD). Belirtilmezse bugün.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Belirli bir konu kümesinin detaylı analizini döndürür:

    - **keywords** — kümeyi tanımlayan anahtar kelimeler
    - **sentiment_distribution** — küme içi pozitif/negatif dağılımı
    - **news** — kümedeki tüm haberler (başlık, kaynak, sentiment, entity, link)

    `cluster_id` değerlerini `/api/today` endpoint'indeki `top_clusters` listesinden alabilirsiniz.
    """
    target = date_str or date.today().isoformat()
    items = _load_analyzed(target)
    clusters = _load_clusters(target)

    cluster_meta = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
    if cluster_meta is None:
        raise HTTPException(status_code=404, detail=f"cluster_id={cluster_id} bulunamadı.")

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
        description="Analiz tarihi (YYYY-MM-DD). Belirtilmezse bugün.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Her haber kaynağı için sentiment istatistiklerini karşılaştırır:

    - **total** — o kaynaktan toplam haber sayısı
    - **sentiment_counts** — pozitif/negatif ham sayılar
    - **sentiment_percentages** — yüzde dağılımı (0-100)
    - **avg_confidence** — modelin ortalama güven skoru

    Hangi kaynağın daha olumsuz haber ürettiğini analiz etmek için kullanılır.
    """
    target = date_str or date.today().isoformat()
    items = _load_analyzed(target)

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
                round(sum(data["scores"]) / len(data["scores"]), 4) if data["scores"] else None
            ),
        }

    return {"date": target, "sources": result}
