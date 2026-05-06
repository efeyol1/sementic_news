# Semantic News

Türkiye'nin 10 büyük haber kaynağından RSS ile günlük veri toplayıp sentiment analizi, NER, konu kümeleme ve drift monitoring yapan tam MLOps projesi.

> **V1 (stable, Turkey-only) — pilot ülke modülü.** Proje şu an V2'ye evrilmekte: çok ülkeli Avrupa medya analiz platformu. Türkiye yeni feature'ların stabilize edildiği baseline kalır; ülke konfigürasyonu, generic pipeline, country-aware schema ve framing analizi için bkz. [V2 roadmap](#v2-roadmap-multi-country-european-media-intelligence).

## Mevcut Durum

| Blok | İçerik | Durum |
|------|--------|-------|
| **Pipeline** | RSS → Preprocess → Sentiment → NER → Clustering → Embedding → Drift | ✅ V1 |
| **Veritabanı** | PostgreSQL (Neon) + pgvector + Alembic migrations | ✅ V1 |
| **API** | FastAPI, Pydantic v2, Prometheus drift gauges, 10 endpoint | ✅ V1 |
| **Dashboard** | Next.js, Vercel deploy, trend + cluster grid | ✅ V1 |
| **CI/CD** | GitHub Actions lint+test, Render auto-deploy | ✅ V1 |
| **MLOps** | MLflow tracking, weekly retrain, ONNX int8 inference, behavioral CheckList | ✅ V1 |
| **V2 — Multi-country** | Country YAML config, generic pipeline, framing analysis | 🚧 in progress |

## Özellikler

- **10 Türk haber kaynağı** — Habertürk, Hürriyet, NTV, CNN Türk, Sözcü, Milliyet, Sabah, TRT Haber, Cumhuriyet, Yeni Şafak
- **Zero-shot sentiment** — `joeddav/xlm-roberta-large-xnli` ile çok dilli destek (TR/DE/FR/ES/EN)
- **Named Entity Recognition** — PER / ORG / LOC (`savasy/bert-base-turkish-ner-cased`)
- **Konu kümeleme** — Sentence-transformer embedding + KMeans, 15 küme
- **Semantik arama** — pgvector cosine similarity, `/api/similar` endpoint
- **PostgreSQL** — Neon hosted, tüm pipeline verisi kalıcı
- **MLflow** experiment tracking + model registry

## Inference Benchmark — Türkçe Sentiment (CPU)

Fine-tuned BERT (`efeyol11/bert-turkish-sentiment`), `max_length=128`, 100 prediction × 3 backend, Apple M-serisi CPU.

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | QPS | Speedup vs PyTorch | PyTorch ile uyum |
|---|---|---|---|---|---|---|
| PyTorch   | 10.74 | 10.87 | 11.05 | 93.0  | 1.00× | _baseline_ |
| ONNX fp32 | 4.84  | 5.05  | 5.09  | 214.1 | 2.15× | 100.0% |
| ONNX int8 | 2.47  | 2.86  | 2.97  | 411.0 | **3.80×** | 100.0% |

ONNX export + dynamic int8 quantization HuggingFace `optimum` ile yapılıyor; benchmark script (`scripts/benchmark_inference.py`) CI gate olarak ≥%99 uyumu zorunlu kılıyor.

```bash
pip install -e ".[serving]"
python -m src.serving.onnx_export
python scripts/benchmark_inference.py
```

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

## V2 Roadmap — Multi-Country European Media Intelligence

V1 (Turkey-only) pilot ülke modülü olarak stabil. V2'de proje çok ülkeli Avrupa medya analiz platformuna evriliyor — Almanya, Polonya, Fransa, Yunanistan gibi ülkelerde göç / AB politikaları / ulusal kimlik / din-laiklik / aile politikası / ekonomi gibi konuların **media framing pattern**'lerini ölçmek için.

**Etik kural**: "Country X Catholic'tir" gibi basitleştirici claim'ler değil, observable framing pattern'leri. Doğru yorum: _"Migration konusunda seçilen Polonya kaynakları national-identity ve religious-moral framing'i humanitarian framing'e göre daha yoğun kullanıyor"_. Yanlış: _"Polonya medyası Catholic'tir"_.

**Future framing dimensions**: Sentiment'ten ayrı bir `framing` adımı olarak observable frame yoğunlukları ölçülecek. İlk aday frame seti: `religious_moral_frame`, `secular_institutional_frame`, `liberal_rights_frame`, `national_identity_frame`, `security_order_frame`, `economic_cost_frame`, `humanitarian_frame`. Planlanan pipeline yeri: `collect → article_fetch → preprocess → sentiment → framing → NER → clustering → vector_store`.

**Mimari ilke**: Country-specific bilgi **asla** Python kodunda hardcoded olamaz; sadece `configs/countries/*.yaml`. Yeni ülke eklemek = YAML oluştur + RSS source ekle + `python -m src.pipeline --country <name>`.

**Phase 2 done (2026-05-01)**: `configs/countries/turkey.yaml` aktif, `src/config/country_loader.py` slug + ISO code lookup'unu destekler, multi-feed sources (`urls: [list]`) normalize edilir, `rss_collector.py` artık YAML'dan okuyor — RSS_FEEDS sabiti yok. 10 yeni unit test loader'ı validate ediyor.

**Source coverage 8.5× artışı (2026-05-02)**: 10 kaynakta multi-category RSS aggregation + 2 kaynakta Google News sitemap handler + 1 kaynakta HTML sitemap scraping. Toplam günlük unique corpus **475 → 4,032 item**. Per-source kazanımlar: Cumhuriyet 100→864 (8.6×), Habertürk 100→718 (7.2×, RSS+sitemap), Hürriyet 75→680 (9.1×), Sözcü 50→523 (10.5×, sitemap), CNN Türk 35→353 (10×), Milliyet 20→315 (15.8×), Yeni Şafak 15→187 (12.5×, html_sitemap), NTV 20→157 (7.9×), Sabah 10→144 (14.4×), TRT Haber 50→91 (1.8×).

**3 source type**: 
- `rss` — feedparser, default
- `googlenews_sitemap` — Google News namespace-aware sitemap parse, title + tarih inline (Sözcü, Habertürk)
- `html_sitemap` — plain sitemap → URL listesi → ThreadPool ile her article'ın HTML'inden `<title>` + `og:description` çek (Yeni Şafak; ~200 article ~50s)

Aynı outlet farklı type'larla listelenince (örn. Habertürk RSS + Habertürk googlenews_sitemap) DB'de `ON CONFLICT(link)` üzerinden dedupe.

**Phase rollout**:

| # | Phase | Status |
|---|-------|--------|
| 0 | Repository audit | ✅ done |
| 1 | Stabilize Turkey baseline | ✅ done |
| 2 | Country configuration system (`configs/countries/turkey.yaml`) | ✅ done |
| 3 | Country-aware pipeline (`--country` CLI arg) | pending |
| 4 | Country-aware DB schema (`country_code`, `language` columns) | pending |
| 5 | Country-aware FastAPI endpoints (`?country=TR`, `/api/countries`) | pending |
| 6 | Dashboard country selector | pending |
| 7 | Add Germany (first non-TR) | pending |
| 8 | Multi-country GitHub Actions matrix | pending |
| 9 | Per-country pipeline run reports | pending |
| 10 | Framing analysis foundation | pending |
| 11 | Evaluation & quality control | pending |
| 12 | README + presentation refresh | pending |

V2 her phase ayrı branch (`v2/phase-N-slug`) + main'e merge sonrası tag (`v2-phase-N`).

### Next Steps — Article Body Quality & V2

1. **Article parser quality report** — kaynak bazında `parsed / empty / fetch_error / parse_error` oranlarını çıkar; hangi kaynakta generic parser iyi/kötü çalışıyor netleşsin.
2. **Source-specific parser improvements** — generic parser zayıf kalan kaynaklara özel selector ekle; öncelik düşük parse başarısı veya stratejik kaynaklar (örn. Sözcü, TRT Haber, Yeni Şafak, Sabah).
3. **Controlled article backfill** — tek tarih için `--article-limit 100 → 300 → 500` şeklinde kademeli ilerle; her artışta parse status, süre ve DB etkisini ölç.
4. **Body-aware analysis QA** — body ile clustering daha anlamlı mı, sentiment dağılımı aşırı kayıyor mu, source comparison mantıklı kalıyor mu kontrol et; gerekirse body snippet limitlerini ayarla.
5. **Daily pipeline rollout** — kalite yeterliyse production daily pipeline'da önce küçük limit ile `--fetch-articles --article-limit 300` aç, sonra ölçerek artır.
6. **Country-aware DB/pipeline** — `country_code`, `language`, `src.pipeline --country <slug>` ve ardından ilk non-TR ülke.
7. **Framing foundation** — sentiment'ten ayrı `src/analysis/framing.py`; frame dağılımlarını ülke/kaynak/kategori bazında ölç, API/dashboard katmanına daha sonra taşı.

## Telif ve Veri Notu

**Bu repo yalnızca kaynak kodu içerir, haber verisi içermez.**
Kullanıcı kendi kurulumunda `rss_collector.py` ile RSS üzerinden veri toplar.
Toplanan haberlerin telif hakkı ilgili yayın kuruluşlarına aittir.

## Lisans

[MIT](LICENSE)
