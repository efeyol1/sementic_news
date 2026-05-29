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
| **V2 — Multi-country omurga** | Country YAML, country-aware DB/pipeline/API/dashboard, Germany pilot | ✅ Phase 0–7 |
| **V2 — Operasyon & framing** | CI matrix, per-country run reports, framing analizi, quality gate | 🚧 Phase 8–12 |

## Özellikler

- **10 Türk haber kaynağı** — Habertürk, Hürriyet, NTV, CNN Türk, Sözcü, Milliyet, Sabah, TRT Haber, Cumhuriyet, Yeni Şafak
- **Zero-shot sentiment** — `joeddav/xlm-roberta-large-xnli` ile çok dilli destek (TR/DE/FR/ES/EN)
- **Named Entity Recognition** — PER / ORG / LOC (`savasy/bert-base-turkish-ner-cased`)
- **Konu kümeleme** — Sentence-transformer embedding + KMeans, 15 küme
- **Semantik arama** — pgvector cosine similarity, `/api/similar` endpoint
- **Entity framing** — cross-country collocation profilleri (PMI / log-likelihood); dashboard'da entity arama + ülkeler arası karşılaştırma (`/entities`, `/entity/{ref}`)
- **Frame bridge** — entity collocate'lerinden 6-çerçeve yoğunluğu (economic / security / identity / governance / humanitarian / conflict) seed-word lexicon ile (`configs/frames/<lang>.yaml`); dashboard entity sayfasında radar paneli. `(entity, ülke)` scope'unda, ülke-geneli iddiası değil
- **PostgreSQL** — Neon hosted, tüm pipeline verisi kalıcı
- **MLflow** experiment tracking + model registry

## Inference Benchmark — Türkçe Sentiment (CPU)

Fine-tuned BERT (`efeyol11/bert-turkish-sentiment`), 100 prediction × 3 backend, Apple M-serisi CPU.

> **Not**: Aşağıdaki tablo `max_length=128` ölçümleridir. Pipeline body coverage için `max_length=256`'ya geçti; CPU latency yaklaşık 2× artar (re-benchmark Phase 11'de yapılacak). Backend-arası **göreceli** speedup (PyTorch ↔ ONNX) ve PyTorch uyum (%100) değişmez.

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
| `GET /api/entities` | Entity dizini / arama (canonical + QID) |
| `GET /api/entity/{ref}/profile` | Entity'nin ülke profili (collocation + PMI/LLR + frame yoğunluğu) |
| `GET /api/entity/{ref}/compare` | Aynı entity'nin ülkeler arası karşılaştırması |
| `GET /api/entity/{ref}/timeline` | Entity'nin günlük mention hacmi + salience eğrisi |
| `GET /metrics` | Prometheus metrikleri |

> Entity endpoint'lerinde `{ref}` bir Wikidata QID (`Q22686`) ya da canonical
> isimdir; tümü `?country=` (default `turkey`) parametresini destekler, `compare`
> ise doğası gereği çok-ülkelidir (opsiyonel `?countries=DE,FR`).

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

### Phase rollout — durum tablosu

| # | Phase | Status | Tag/PR |
|---|-------|--------|--------|
| 0 | Repository audit | ✅ done | — |
| 1 | Stabilize Turkey baseline | ✅ done | — |
| 2 | Country configuration system (`configs/countries/turkey.yaml`) | ✅ done | `v2-phase-2` |
| 4 | Country-aware DB schema (`country_code`, `language`) | ✅ done | PR #3 (`43d7eda`) |
| 3 | Country-aware pipeline (`--country` CLI arg) | ✅ done | PR #4 (`99e952d`) |
| 5 | Country-aware FastAPI endpoints (`?country=TR`, `/api/countries`) | ✅ done | PR #5 (`b779155`) |
| 6 | Dashboard country selector + `?country` propagation | ✅ done | PR #6 (`bbfffce`) |
| 7 | Add Germany (first non-TR) | ✅ done | `germany.yaml` (`2026-05-14`) |
| 8 | Multi-country GitHub Actions matrix | 🟡 next | — |
| 9 | Per-country pipeline run reports | pending | — |
| 10 | Framing analysis foundation | pending | — |
| 11 | Evaluation & quality control (gold seed + accuracy gate) | 🟡 partial — DE gold seed in progress | `data/qa/sentiment_review_germany_*.csv` |
| 12 | README + presentation refresh | pending | — |

V2 omurga (Phase 0–7) tamamlandı: yeni bir ülkeyi eklemek için **sadece** `configs/countries/<slug>.yaml` oluşturmak yetiyor; kod değişikliği gerekmiyor. Kalan iş operasyonel sertleştirme (CI matrix, run reports) + ürün katmanı (framing) + kalite ölçümü.

Git policy: her phase ayrı branch (`v2/phase-N-slug`), main'e PR + merge, sonra `v2-phase-N` tag.

### Next Steps — Phase 8'den itibaren

1. **Phase 8 — Multi-country CI matrix.** `daily_pipeline.yml` şu an tek country ile çalışıyor (ima edilen TR default). `strategy.matrix.country: [turkey, germany]` ekle; her job kendi `--country` arg'ı ile koşsun. Bir ülke fail olursa diğeri kararsız kalmasın. Geçiş kriteri: 3 gün üst üste her iki country için run yeşil + dashboard'da DE verisi görünür.
2. **Phase 9 — Per-country run reports.** `data/reports/{slug}/{date}_run_report.json` ile her pipeline koşusu için yapısal özet (sources_success/failed, articles_collected, sentiment_analyzed, duration_seconds, errors). `src/pipeline.py` sonunda yaz, failure path'inde de mümkünse yaz. Bu, drift dashboard'unun yanına "operational health" görünürlüğü ekler.
3. **Phase 10 — Framing analizi foundation.** `src/analysis/framing.py` placeholder modülü + future-ready frame skor kolonu (JSONB). İlk aday frame seti: `religious_moral`, `secular_institutional`, `liberal_rights`, `national_identity`, `security_order`, `economic_cost`, `humanitarian`. **Henüz model build etme** — sadece schema, frame taxonomy dokümante, observable-pattern dili kullanma kuralları (etik bölüm).
4. **Phase 11 — Quality gate.** TR ve DE için gold seed → `accuracy` / `macro_f1` / `neutral_recall` ölçümü. Şu an `scripts/build_sentiment_gold_dataset.py` + `train_reviewed_sentiment.py` + Almanya gold seed (`data/qa/sentiment_review_germany_*.csv`) hazır; eksik olan accuracy gate'in retraining pipeline'a bağlanması (regression olursa model push iptal). Detay alt başlık → [Sentiment QA Workflow](#sentiment-qa-workflow).
5. **Phase 12 — README + presentation.** Tam rewrite: long-term vision, framing methodology, ethics section, CV/internship-ready görünüm. Phase 10 + 11 sonuçları elde olduktan sonra.

### Yapısal borçlar (Phase'lara paralel)

- **Clustering iyileştirmesi**: KMeans k=15 sabit, cluster'lar keskin değil. Body artık embedding'de kullanılıyor ama somut iyileştirme ölçülmedi. Öneri sırası: UMAP + dinamik k → HDBSCAN → BERTopic. Phase 10 öncesi karar verilmeli (cluster'lar framing analizinin girdisi). Bkz. memory `project_clustering_plan.md`.
- **Sentiment QA country-aware refactor**: Audit/export script'leri Phase 4 sonrası `--country` arg destekliyor ama cue listesi hâlâ TR'ye gömülü (`audit_sentiment_quality.py`). DE için aynı altyapı çalışır ama Türkçe cue'lar DE corpus'unda anlamsız sinyal verir; ya cue'ları YAML'a taşı, ya `src/analysis/cues/{language}.py` ile modülerleştir.
- **Dashboard UX**: DatePicker `?country` query param'ını koruyor mu test edilmedi (bilinen olası bug). API 404 yerine boş response döndürmesi UX kararı — şu an "veri bulunamadı" mesajı tüm boş durumları aynı gösteriyor.

### Beyond V2 — Olası V3 / Alternatif Yönler

V2 tamamlandığında platform "çok ülkeli framing dashboard". Buradan iki ana ileri yön var; karar V2 Phase 12 sonrası user/research need'e göre verilir:

**Yön A — Derinleşme (V3a: Analysis depth)**
- Custom-trained framing classifier (Phase 10 placeholder yerine gerçek model)
- Cross-country event tracking: aynı olayı (örn. AB göç anlaşması) farklı ülke medyalarında karşılaştır
- Temporal narrative shift detection (story arc evolution over weeks)
- Source bias modeling (kaynak başına framing yoğunluk imzası)
- LLM-based explainability: belirli bir cluster için "şu ülke medyası bu olayı `national_identity` frame'i ile çerçeveliyor, kanıt cümleler: ..." raporu

**Yön B — Yayılma (V3b: Coverage breadth)**
- 5+ AB ülkesi (PL, FR, GR, IT, ES)
- Sosyal medya ekleme (Twitter/X public posts via Academic API, eğer hâlâ erişilebilir; veya Bluesky)
- TV haberlerinin transcript'lerinden ingestion (YouTube/podcast feed)
- Multi-modal: görsel framing (haber fotoğraflarından gözlemlenebilir kompozisyon imzaları — risky, etik dikkat)
- Public API + access tier: araştırmacılar için ücretsiz read-only access

**Yön C — Ürünleşme (V3c: Product)**
- Newsletter çıktısı: haftalık framing özeti (e-mail otomasyonu)
- Embeddable widget: araştırmacıların kendi sitelerine takabileceği framing chart
- Alerts: belirli bir frame'in yoğunluğu eşiği aşınca bildirim
- B2B: gazete redaksiyonları / akademik kurumlar için detaylı pano

**Karar kriteri**: V2 Phase 12 bittikten sonra hangi yönün **gerçek bir kullanıcı problemini** çözdüğüne bak. Yön A internship/akademik portföy için en güçlü; Yön B veriye doğal genişleme; Yön C ticari iz. Karışım da mümkün ama büyüklüğünden ötürü genelde tek yön seçmek hızlıdır.

Önemli: V3 yönü ne olursa olsun, mimari ilke aynı kalır — **country-specific ya da source-specific bilgi asla Python koduna gömülmez, YAML / config / data layer'da kalır**. Bu kural V2'de korundu, V3'te de korunmalı.

### Sentiment QA Workflow

Body parsing sonrası model kalitesini doğrulamak için sentiment skorları önce freshness ve dağılım açısından audit edilir:

```bash
python scripts/audit_sentiment_quality.py --date 2026-05-06
python -m src.analysis.sentiment --date 2026-05-06 --only-missing --limit 200
python -m src.analysis.sentiment --date 2026-05-06 --only-stale-after-body --limit 100
python scripts/export_sentiment_review_sample.py --date 2026-05-06 --per-source-label 2
python scripts/evaluate_sentiment_review.py data/qa/sentiment_review_2026-05-06.csv
```

Audit'te `stale_after_body`, düşük neutral oranı ve yüksek güvenli bariz cue mismatch örnekleri izlenir. Model kalitesi için asıl gate, manuel doldurulmuş `reviewed_label` üzerinden `accuracy`, `macro_f1` ve özellikle `neutral recall` değerleridir.

`--only-with-body` flag'i body'si olan satırları **mevcut skorlara bakmadan yeniden yazar**; yalnızca günü kasten baştan rescore ederken kullan, normal akışta `--only-missing` ve `--only-stale-after-body` yeterli.

### Germany Production Pilot Runbook

Almanya için Türkçe fine-tuned / ONNX sentiment env'leri kapalı olmalı; bu koşul sağlanmazsa pipeline fail-fast davranır:

```bash
unset SENTIMENT_MODEL_ID SENTIMENT_BACKEND SENTIMENT_ONNX_DIR
python -m src.pipeline --country germany --date YYYY-MM-DD
```

Article body fetch ile pilot koşmak için:

```bash
unset SENTIMENT_MODEL_ID SENTIMENT_BACKEND SENTIMENT_ONNX_DIR
python -m src.pipeline --country germany --date YYYY-MM-DD --fetch-articles --article-limit 100
```

Neon bağlantısı `select 1` aşamasında bile `SSL SYSCALL` / `Connection reset by peer`
ile düşüyorsa pipeline'ı tekrar denemeden önce Neon status ve console kontrol edilmeli.
Collect insert kısmen başarılı olduysa Almanya pilot devam komutu:

```bash
unset SENTIMENT_MODEL_ID SENTIMENT_BACKEND SENTIMENT_ONNX_DIR
python -m src.pipeline --country germany --date YYYY-MM-DD --skip-collect --fetch-articles --article-limit 100
```

QA komutları country-aware çalışır; default hâlâ `turkey`:

```bash
python scripts/analyze_article_parse_quality.py --country germany --date YYYY-MM-DD
python scripts/audit_sentiment_quality.py --country germany --date YYYY-MM-DD
python scripts/export_sentiment_review_sample.py --country germany --date YYYY-MM-DD --per-source-label 2
```

## Telif ve Veri Notu

**Bu repo yalnızca kaynak kodu içerir, haber verisi içermez.**
Kullanıcı kendi kurulumunda `rss_collector.py` ile RSS üzerinden veri toplar.
Toplanan haberlerin telif hakkı ilgili yayın kuruluşlarına aittir.

## Lisans

[MIT](LICENSE)
