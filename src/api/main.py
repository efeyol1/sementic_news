"""FastAPI application for Semantic News TR."""

from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.api import drift_metrics  # noqa: F401  registers Prometheus collector on import
from src.api.dependencies import resolve_country
from src.config.country_loader import list_available_countries
from src.db.queries import (
    fetch_all_for_api,
    fetch_available_dates,
    fetch_cluster_summaries,
    fetch_drift_history,
    fetch_latest_drift_report,
    fetch_sentiment_trend,
    fetch_top_entities,
)
from src.db.schema import init_db

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail-soft: a transient DB hiccup at boot shouldn't keep the API
    # offline — the daily pipeline / Render deploy hook already runs
    # migrations, so missing them here is recoverable on the next request.
    try:
        init_db()
    except Exception as exc:
        logger.warning(f"DB init on startup failed: {exc}")
    yield


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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class SentimentCounts(BaseModel):
    positive: int = Field(..., examples=[264])
    negative: int = Field(..., examples=[276])
    neutral: int = Field(0, examples=[120])


class SentimentPercentages(BaseModel):
    positive: float = Field(..., examples=[48.9])
    negative: float = Field(..., examples=[51.1])
    neutral: float = Field(0.0, examples=[18.2])


class SentimentSummary(BaseModel):
    counts: SentimentCounts
    percentages: SentimentPercentages


class ClusterSummary(BaseModel):
    cluster_id: int = Field(..., examples=[3])
    title: str = Field(..., examples=["Ekonomi · Merkez Bankası"])
    size: int = Field(..., examples=[47])
    keywords: list[str] = Field(..., examples=[["ekonomi", "dolar", "faiz"]])


class ClusterDetail(BaseModel):
    """Full per-cluster stats — used by the dashboard's cluster grid."""
    cluster_id: int = Field(..., examples=[3])
    title: str = Field(..., examples=["Ekonomi · Merkez Bankası"])
    size: int = Field(..., examples=[47])
    keywords: list[str] = Field(..., examples=[["ekonomi", "dolar", "faiz", "merkez", "banka"]])
    sources: dict[str, int] = Field(
        ..., examples=[{"Hürriyet": 12, "Cumhuriyet": 8, "Habertürk": 7}]
    )
    sentiment_distribution: dict[str, int] = Field(
        ..., examples=[{"positive": 5, "neutral": 30, "negative": 12}]
    )


class ClustersResponse(BaseModel):
    date: str = Field(..., examples=["2026-04-20"])
    total_clusters: int = Field(..., examples=[15])
    clusters: list[ClusterDetail]


class TopEntities(BaseModel):
    PER: list[str] = Field(..., examples=[["Erdoğan", "Trump"]])
    ORG: list[str] = Field(..., examples=[["TBMM", "Merkez Bankası"]])
    LOC: list[str] = Field(..., examples=[["Ankara", "İstanbul"]])


class TodayResponse(BaseModel):
    date: str = Field(..., examples=["2026-04-20"])
    total_items: int = Field(..., examples=[545])
    turkish_items: int = Field(..., examples=[540])
    sources: dict[str, int] = Field(..., examples=[{"Cumhuriyet": 114}])
    sentiment: SentimentSummary
    top_entities: TopEntities
    cluster_count: int = Field(..., examples=[15])
    top_clusters: list[ClusterSummary]
    available_dates: list[str] = Field(..., examples=[["2026-04-20"]])


class NewsItem(BaseModel):
    title: str = Field(..., examples=["Kabine toplantısı ne zaman?"])
    source_name: str = Field(..., examples=["Habertürk"])
    published_date: str = Field(..., examples=["2026-04-20T07:30:00+00:00"])
    sentiment_label: str | None = Field(None, examples=["positive"])
    sentiment_score: float | None = Field(None, examples=[0.977])
    calibrated_sentiment_score: float | None = Field(None, examples=[0.812])
    entities: dict[str, list[str]] | None = Field(None)
    link: str | None = Field(None)


class TopicResponse(BaseModel):
    date: str = Field(..., examples=["2026-04-20"])
    cluster_id: int = Field(..., examples=[3])
    keywords: list[str]
    size: int = Field(..., examples=[47])
    sentiment_distribution: dict[str, int]
    news: list[NewsItem]


class SourceStats(BaseModel):
    total: int = Field(..., examples=[109])
    sentiment_counts: dict[str, int]
    sentiment_percentages: dict[str, float]
    avg_confidence: float | None = Field(None, examples=[0.861])


class SourceComparisonResponse(BaseModel):
    date: str = Field(..., examples=["2026-04-20"])
    sources: dict[str, SourceStats]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class SimilarNewsItem(BaseModel):
    title: str
    source_name: str
    date: str
    sentiment_label: str
    link: str | None
    similarity: float = Field(..., examples=[0.91])


class SimilarNewsResponse(BaseModel):
    query_title: str
    results: list[SimilarNewsItem]


class DatesResponse(BaseModel):
    dates: list[str] = Field(..., examples=[["2026-04-20", "2026-04-21"]])


class TrendPoint(BaseModel):
    date: str = Field(..., examples=["2026-04-20"])
    positive: int
    negative: int
    neutral: int = 0
    total: int
    positive_pct: float


class TrendResponse(BaseModel):
    points: list[TrendPoint]


class CountryInfo(BaseModel):
    code: str = Field(..., examples=["TR"])
    slug: str = Field(..., examples=["turkey"])
    name: str = Field(..., examples=["Turkey"])
    language: str = Field(..., examples=["tr"])
    status: str = Field("active", examples=["active"])


class AboutLinks(BaseModel):
    repository: str
    license: str
    model_card: str
    data_provenance: str
    data_license: str
    privacy: str
    contributing: str
    code_of_conduct: str
    security: str


class AboutResponse(BaseModel):
    name: str = Field(..., examples=["semantic-news"])
    version: str = Field(..., examples=["2.0.0"])
    disclaimer: str
    disclaimer_tr: str
    links: AboutLinks


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _get_items(date_str: str, country_code: str = "TR") -> list[dict]:
    items = fetch_all_for_api(date_str, country_code=country_code)
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} tarihine ait analiz verisi bulunamadı.",
        )
    return items


def _get_clusters(date_str: str, country_code: str = "TR") -> list[dict]:
    clusters = fetch_cluster_summaries(date_str, country_code=country_code)
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


_REPO_URL = "https://github.com/efeyol11/sementic_news"
_BLOB_BASE = f"{_REPO_URL}/blob/main"


@app.get(
    "/about",
    tags=["Sistem"],
    summary="Proje meta, disclaimer ve governance link'leri",
    response_model=AboutResponse,
)
def about():
    """Public-release disclaimer + governance doc'larına link'ler.

    Bu endpoint kasıtlı olarak DB'ye dokunmaz — repo public olduğu
    için audit ve compliance araçları statik bir endpoint'ten lisans /
    veri politikası bilgisine ulaşabilsin diye eklendi.
    """
    return {
        "name": "semantic-news",
        "version": app.version,
        "disclaimer": (
            "Outputs are descriptive signals derived from publicly available "
            "news content. They are not investment advice, risk ratings, "
            "factual guarantees, or editorial endorsements."
        ),
        "disclaimer_tr": (
            "Çıktılar, kamuya açık haber içeriğinden türetilmiş betimleyici "
            "sinyallerdir. Yatırım tavsiyesi, risk derecelendirmesi, "
            "doğruluk garantisi veya editöryal onay niteliği taşımaz."
        ),
        "links": {
            "repository": _REPO_URL,
            "license": f"{_BLOB_BASE}/LICENSE",
            "model_card": f"{_BLOB_BASE}/MODEL_CARD.md",
            "data_provenance": f"{_BLOB_BASE}/DATA_PROVENANCE.md",
            "data_license": f"{_BLOB_BASE}/DATA_LICENSE.md",
            "privacy": f"{_BLOB_BASE}/PRIVACY.md",
            "contributing": f"{_BLOB_BASE}/CONTRIBUTING.md",
            "code_of_conduct": f"{_BLOB_BASE}/CODE_OF_CONDUCT.md",
            "security": f"{_BLOB_BASE}/SECURITY.md",
        },
    }


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
    country_config: dict = Depends(resolve_country),
):
    target = date_str or date.today().isoformat()
    cc = country_config["country_code"]
    items = _get_items(target, country_code=cc)
    clusters = fetch_cluster_summaries(target, country_code=cc)

    turkish = [i for i in items if i.get("is_turkish")]
    n = len(turkish)

    _counts = Counter(i.get("sentiment_label") for i in turkish if i.get("sentiment_label"))
    sentiment_counts = {
        "positive": _counts.get("positive", 0),
        "negative": _counts.get("negative", 0),
        "neutral":  _counts.get("neutral", 0),
    }
    sentiment_pct = (
        {k: round(v / n * 100, 1) for k, v in sentiment_counts.items()}
        if n else {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    )

    top_entities = fetch_top_entities(target, country_code=cc)
    if not any(top_entities.values()):
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
        "available_dates": fetch_available_dates(country_code=cc),
    }


@app.get(
    "/api/clusters",
    tags=["Analiz"],
    summary="Tüm cluster'lar (15) ve istatistikleri",
    response_model=ClustersResponse,
    responses={404: {"description": "Belirtilen tarihe ait cluster verisi bulunamadı"}},
)
def all_clusters(
    date_str: str = Query(
        default=None,
        alias="date",
        description="Analiz tarihi (YYYY-MM-DD). Belirtilmezse bugünün verisi döner.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    country_config: dict = Depends(resolve_country),
):
    """Dashboard cluster grid için 15 cluster'ı zengin stats'larla döner.

    Drill-down için ``/api/topic/{cluster_id}`` kullanılır — bu endpoint
    sadece liste/grid view içindir. Cluster'lar size DESC sıralı.
    """
    target = date_str or date.today().isoformat()
    clusters = _get_clusters(target, country_code=country_config["country_code"])
    return {
        "date": target,
        "total_clusters": len(clusters),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "title": c.get("title") or (c["keywords"][0] if c["keywords"] else ""),
                "size": c["size"],
                "keywords": c["keywords"],
                "sources": c.get("sources") or {},
                "sentiment_distribution": c.get("sentiment_distribution") or {},
            }
            for c in clusters
        ],
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
    country_config: dict = Depends(resolve_country),
):
    target = date_str or date.today().isoformat()
    cc = country_config["country_code"]
    items = _get_items(target, country_code=cc)
    clusters = fetch_cluster_summaries(target, country_code=cc)

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
            "calibrated_sentiment_score": i.get("calibrated_sentiment_score"),
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
    country_config: dict = Depends(resolve_country),
):
    target = date_str or date.today().isoformat()
    items = _get_items(target, country_code=country_config["country_code"])

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
        score = item.get("calibrated_sentiment_score")
        if score is None:
            score = item.get("sentiment_score")
        if score is not None:
            sources[src]["scores"].append(score)

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
    "/api/dates",
    tags=["Analiz"],
    summary="Mevcut analiz tarihleri",
    response_model=DatesResponse,
)
def available_dates(country_config: dict = Depends(resolve_country)):
    return {"dates": fetch_available_dates(country_code=country_config["country_code"])}


@app.get(
    "/api/trend",
    tags=["Analiz"],
    summary="Çok günlük sentiment trendi",
    response_model=TrendResponse,
)
def sentiment_trend(
    days: int = Query(default=30, ge=7, le=90, description="Kaç günlük veri"),
    country_config: dict = Depends(resolve_country),
):
    rows = fetch_sentiment_trend(days, country_code=country_config["country_code"])
    points = [
        {
            "date": r["date"],
            "positive": r["positive"],
            "negative": r["negative"],
            "neutral": r.get("neutral", 0) or 0,
            "total": r["total"],
            "positive_pct": round(r["positive"] / r["total"] * 100, 1) if r["total"] else 0.0,
        }
        for r in rows
    ]
    return {"points": points}


@app.get(
    "/api/similar",
    tags=["Analiz"],
    summary="Semantik olarak benzer haberler",
    response_model=SimilarNewsResponse,
    responses={
        404: {"description": "Vektör veritabanı boş veya haber bulunamadı"},
        503: {"description": "Semantik arama bu deployment'ta kapalı (ML stack yüklü değil)"},
    },
)
def similar_news(
    q: str = Query(..., description="Aranacak haber başlığı veya metin", min_length=5),
    n: int = Query(default=5, ge=1, le=20),
    country_config: dict = Depends(resolve_country),
):
    # ML stack (torch/sentence-transformers) lives in the `pipeline` extra,
    # not in the API's base install — see pyproject.toml. On the lean
    # Render deployment the import/encode raises ImportError; surface it as
    # a 503 instead of a 500 so the dashboard can degrade gracefully.
    try:
        from src.analysis.vector_store import find_similar

        results = find_similar(q, n=n, country_code=country_config["country_code"])
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Semantik arama bu deployment'ta kapalı — ML bağımlılıkları yüklü değil.",
        )
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


@app.get(
    "/api/drift/latest",
    tags=["Sistem"],
    summary="En son drift raporu (PSI + per-class delta)",
    responses={404: {"description": "Henüz drift raporu üretilmedi"}},
)
def latest_drift(country_config: dict = Depends(resolve_country)):
    row = fetch_latest_drift_report(country_code=country_config["country_code"])
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Henüz drift raporu yok — daily pipeline'ın drift step'inin çalışmasını bekleyin.",
        )
    return row


@app.get(
    "/api/drift/history",
    tags=["Sistem"],
    summary="Drift raporları zaman serisi",
)
def drift_history(
    days: int = Query(default=30, ge=7, le=180, description="Kaç günlük geriye"),
    country_config: dict = Depends(resolve_country),
):
    return {"reports": fetch_drift_history(days, country_code=country_config["country_code"])}


@app.get(
    "/api/countries",
    tags=["Sistem"],
    summary="Yapılandırılmış ülkelerin listesi",
    response_model=list[CountryInfo],
)
def list_countries():
    """``configs/countries/*.yaml``'dan okunan tüm ülkeler.

    Dashboard country selector'ı bu endpoint'i çağırır (Phase 6). Yeni
    ülke eklemek = ``configs/countries/<slug>.yaml`` dosyası eklemek;
    Python kodu değişmez.
    """
    return list_available_countries()
