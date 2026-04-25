# Semantic News TR

Türkiye'nin 10 büyük haber kaynağından RSS ile günlük veri toplayıp sentiment analizi, NER ve konu kümeleme yapan tam MLOps projesi.

## Mevcut Durum

| Blok | İçerik | Durum |
|------|--------|-------|
| **Pipeline** | RSS → Preprocess → Sentiment → NER → Clustering → Embedding | ✅ |
| **Veritabanı** | PostgreSQL (Neon) + pgvector semantik arama | ✅ |
| **API** | FastAPI, Pydantic modeller, Prometheus metrikleri | ✅ |
| **Dashboard** | Next.js, Vercel deploy, trend grafikler | ✅ |
| **CI/CD** | GitHub Actions lint+test+deploy, Render CD | ✅ |
| **MLOps** | MLflow experiment tracking, weekly retrain | ✅ |

## Özellikler

- **10 Türk haber kaynağı** — Habertürk, Hürriyet, NTV, CNN Türk, Sözcü, Milliyet, Sabah, TRT Haber, Cumhuriyet, Yeni Şafak
- **Zero-shot sentiment** — `joeddav/xlm-roberta-large-xnli` ile çok dilli destek (TR/DE/FR/ES/EN)
- **Named Entity Recognition** — PER / ORG / LOC (`savasy/bert-base-turkish-ner-cased`)
- **Konu kümeleme** — Sentence-transformer embedding + KMeans, 15 küme
- **Semantik arama** — pgvector cosine similarity, `/api/similar` endpoint
- **PostgreSQL** — Neon hosted, tüm pipeline verisi kalıcı
- **MLflow** experiment tracking + model registry

## Kurulum

```bash
git clone https://github.com/efeyol11/sementic_news.git
cd sementic_news

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

# .env dosyası oluştur
cp .env.example .env
# DATABASE_URL= satırını Neon bağlantı stringiyle doldur

# Veritabanı şemasını kur
python -m src.db.schema
```

## Pipeline

```bash
# Tam zincir (bugün)
python -m src.pipeline

# Belirli tarih
python -m src.pipeline --date 2026-04-25

# RSS zaten çekildiyse atla
python -m src.pipeline --skip-collect

# Adım adım
python -m src.data.rss_collector
python -m src.data.preprocessor
python -m src.analysis.sentiment
python -m src.analysis.ner
python -m src.analysis.clustering
python -m src.analysis.vector_store
```

## API

```bash
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

| Endpoint | Açıklama |
|----------|----------|
| `GET /health` | Liveness probe |
| `GET /api/today` | Günlük analiz özeti |
| `GET /api/topic/{id}` | Cluster haberleri |
| `GET /api/source-comparison` | Kaynak bazlı sentiment |
| `GET /api/similar?q=...` | Semantik benzer haberler (pgvector) |
| `GET /metrics` | Prometheus metrikleri |

## Dashboard

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:3000
```

## Testler

```bash
pytest tests/ -v
ruff check src/ tests/
```

## Deploy

| Servis | Platform | Trigger |
|--------|----------|---------|
| **API** | Render | `git push main` |
| **Dashboard** | Vercel | `git push main` |
| **Daily pipeline** | GitHub Actions | 04:00 UTC (07:00 TR) |
| **Weekly retrain** | GitHub Actions | Pazar 02:00 UTC |

**Gerekli secrets:**
- `DATABASE_URL` — Neon PostgreSQL connection string (GitHub Actions + Render)
- `RENDER_DEPLOY_HOOK_URL` — Render deploy webhook
- `HF_TOKEN` — HuggingFace model push (retrain için)

## Proje Yapısı

```
sementic_news/
├── src/
│   ├── data/
│   │   ├── rss_collector.py    # RSS veri toplama → PostgreSQL
│   │   ├── preprocessor.py     # HTML temizleme, dil tespiti
│   │   └── dataset.py          # HuggingFace dataset (fine-tune için)
│   ├── analysis/
│   │   ├── sentiment.py        # Zero-shot multilingual sentiment
│   │   ├── ner.py              # Named entity recognition
│   │   ├── clustering.py       # TF-IDF + KMeans konu kümeleme
│   │   └── vector_store.py     # pgvector embedding + semantik arama
│   ├── training/
│   │   ├── train.py            # BERT fine-tuning
│   │   ├── retrain.py          # Weekly retrain
│   │   ├── evaluate.py         # Metrik hesaplama
│   │   └── baseline.py         # Baseline karşılaştırma
│   ├── db/
│   │   ├── client.py           # PostgreSQL bağlantı havuzu
│   │   ├── schema.py           # DDL (news_items, cluster_summaries)
│   │   └── queries.py          # Tüm DB sorguları
│   ├── api/
│   │   └── main.py             # FastAPI uygulaması
│   └── pipeline.py             # Günlük zincir orkestratörü
├── dashboard/                  # Next.js 14 App Router
│   └── src/
│       ├── app/                # Sayfalar
│       ├── components/         # UI bileşenleri
│       └── lib/                # API client, utils
├── tests/
│   ├── test_api.py
│   └── test_vector_store.py
├── .github/workflows/
│   ├── deploy.yml              # CI + Render CD
│   ├── daily_pipeline.yml      # Günlük pipeline cron
│   └── weekly_retrain.yml      # Haftalık model yenileme
├── monitoring/                 # Grafana + Prometheus config
├── docker-compose.yml
└── pyproject.toml
```

## Telif ve Veri Notu

**Bu repo yalnızca kaynak kodu içerir, haber verisi içermez.**
Kullanıcı kendi kurulumunda `rss_collector.py` ile RSS üzerinden veri toplar.
Toplanan haberlerin telif hakkı ilgili yayın kuruluşlarına aittir.

## Lisans

[MIT](LICENSE)
