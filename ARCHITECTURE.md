# Semantic News TR — Mimari ve Proje Akışı

## Projenin Özü

10 Türk haber kaynağından her gün otomatik olarak haber çekip bunları üç farklı ML modeliyle analiz eden, sonuçları bir API üzerinden sunan tam bir MLOps pipeline'ı.

---

## Büyük Resim

```
┌─────────────────────────────────────────────────────────────┐
│                        GÜNLÜK DÖNGÜ                         │
│                                                             │
│   RSS Kaynakları                                            │
│   (10 site)  ──►  Collect  ──►  Preprocess  ──►  Analyze   │
│                                                     │       │
│                                              ┌──────┴───┐   │
│                                              │ sentiment│   │
│                                              │   NER    │   │
│                                              │clustering│   │
│                                              └──────┬───┘   │
│                                                     │       │
│                                              data/analyzed/ │
└─────────────────────────────────────────────────────┼───────┘
                                                      │
                                              ┌───────▼──────┐
                                              │  FastAPI     │
                                              │  /api/today  │
                                              │  /api/topic  │
                                              │  /metrics    │
                                              └───────┬──────┘
                                                      │
                                         ┌────────────┴──────────┐
                                         │                       │
                                   ┌─────▼─────┐         ┌──────▼──────┐
                                   │  Grafana  │         │  Next.js    │
                                   │ Dashboard │         │  Dashboard  │
                                   │ (izleme)  │         │  (:3000)    │
                                   └───────────┘         └─────────────┘
```

---

## Pipeline Adımları

### 1. Veri Toplama — `src/data/rss_collector.py`

```
Habertürk ──┐
Hürriyet   ──┤
NTV        ──┤
CNN Türk   ──┼──► feedparser ──► deduplicate ──► data/raw/YYYY-MM-DD.json
Sözcü      ──┤     (link bazlı)
Milliyet   ──┤
Sabah      ──┤
TRT Haber  ──┤
Cumhuriyet ──┤
Yeni Şafak ──┘
```

**Ne yapıyor:** Her kaynaktan XML/Atom feed çeker, girişleri normalize eder, link bazlı tekrar kontrolü yapar, bugünün dosyasına ekler.

**Çıktı alanları:** `title, summary, source_name, published_date, link, category`

---

### 2. Ön İşleme — `src/data/preprocessor.py`

```
data/raw/YYYY-MM-DD.json
         │
         ▼
    HTML temizle  ──► Çok kısaları filtrele (< 10 karakter)
         │
         ▼
    Boşluk normalize  ──► Tarih normalize (ISO-8601)
         │
         ▼
    Dil tespiti (langdetect) ──► is_turkish flag
         │
         ▼
data/processed/YYYY-MM-DD.json
```

**Eklenen alanlar:** `cleaned_title, cleaned_summary, is_turkish, char_count`

---

### 3. Sentiment Analizi — `src/analysis/sentiment.py`

```
data/processed/  ──►  savasy/bert-base-turkish-sentiment-cased
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                      positive   neutral   negative
                         │
                         ▼
                  sentiment_label + sentiment_score + sentiment_scores
                         │
                         ▼
                data/analyzed/YYYY-MM-DD.json  (ilk yazım)
```

**Model:** BERT tabanlı, Türkçe için fine-tune edilmiş. Her haber için 3 sınıf olasılığı döner, en yüksek olan `sentiment_label` olur.

**MLflow:** Her run için `negative_pct`, `positive_pct`, `avg_confidence`, `duration_sec` loglanır.

---

### 4. Named Entity Recognition — `src/analysis/ner.py`

```
data/analyzed/  ──►  savasy/bert-base-turkish-ner-cased
                                │
                     ┌──────────┼──────────┐
                     ▼          ▼          ▼
                    PER        ORG        LOC
               (kişi adı)  (kurum)   (yer adı)
                     │
                     ▼
              entities: {PER: [...], ORG: [...], LOC: [...]}
              entity_count: int
                     │
                     ▼
              data/analyzed/YYYY-MM-DD.json  (üzerine yazar)
```

**Örnek çıktı:**
```json
{
  "entities": {
    "PER": ["Erdoğan", "Özel"],
    "ORG": ["TBMM", "Bakanlık"],
    "LOC": ["Ankara", "İstanbul"]
  },
  "entity_count": 6
}
```

---

### 5. Konu Kümeleme — `src/analysis/clustering.py`

```
Tüm haberler
     │
     ▼
TF-IDF vektörleştirme (5000 feature, unigram+bigram)
     │
     ▼
KMeans (15 cluster, random_state=42)
     │
     ├──► Her habere: cluster_id + cluster_keywords
     │
     └──► data/analyzed/YYYY-MM-DD_clusters.json
          (her cluster için: boyut, keywords, kaynak dağılımı, sentiment dağılımı)
```

**Silhouette score** ile kümeleme kalitesi MLflow'a loglanır.

---

## Veri Şeması — Tam Yaşam Döngüsü

```
Ham (raw)          İşlenmiş (processed)      Analiz edilmiş (analyzed)
─────────          ────────────────────      ─────────────────────────
title          →   title                 →   title
summary        →   summary               →   summary
source_name    →   source_name           →   source_name
published_date →   published_date (ISO)  →   published_date
link           →   link                  →   link
category       →   category              →   category
               →   cleaned_title         →   cleaned_title
               →   cleaned_summary       →   cleaned_summary
               →   is_turkish            →   is_turkish
               →   char_count            →   char_count
                                         →   sentiment_label   ← sentiment.py
                                         →   sentiment_score   ← sentiment.py
                                         →   sentiment_scores  ← sentiment.py
                                         →   analyzed_at       ← sentiment.py
                                         →   entities          ← ner.py
                                         →   entity_count      ← ner.py
                                         →   cluster_id        ← clustering.py
                                         →   cluster_keywords  ← clustering.py
```

---

## MLOps Katmanı

### DVC — Pipeline Versiyonlama

```
dvc repro
    │
    ├── [collect]    → data/raw/
    ├── [preprocess] → data/processed/   (deps: data/raw, preprocessor.py)
    └── [analyze]    → data/analyzed/    (deps: data/processed, sentiment/ner/clustering.py)
```

Kaynak kodunda değişiklik yoksa DVC ilgili stage'i atlar — her gün sadece değişen kısım çalışır.

### MLflow — Experiment Tracking

```
Experiments:
├── turkish-sentiment   ← model eğitimi (train.py)
├── news-sentiment      ← günlük analiz run'ları
├── news-ner            ← günlük NER run'ları
└── news-clustering     ← günlük clustering run'ları

Model Registry:
└── turkish-news-sentiment
    ├── v1 (staging)
    └── v2 (production)  ← F1 > 0.75 şartı
```

```bash
mlflow ui  # http://localhost:5000
```

---

## FastAPI Servisi

```
GET /health
    └── {"status": "ok"}

GET /api/today?date=2026-04-20
    └── {total_items, turkish_items, sentiment{}, top_entities{}, top_clusters[]}

GET /api/topic/{cluster_id}?date=2026-04-20
    └── {keywords[], size, sentiment_distribution{}, news[]}

GET /api/source-comparison?date=2026-04-20
    └── {sources: {kaynak: {total, sentiment_percentages{}, avg_confidence}}}

GET /metrics
    └── Prometheus formatında HTTP metrikleri
```

---

## Monitoring Stack

```
FastAPI (:8000)
    │  /metrics (Prometheus formatı)
    ▼
Prometheus (:9090)
    │  scrape / 15s
    ▼
Grafana (:3000)
    │
    ├── İstek/dakika (endpoint bazlı)
    ├── Ortalama yanıt süresi
    ├── HTTP 5xx hata oranı
    └── Endpoint dağılımı (pie chart)
```

Yerel çalıştırmak için:
```bash
docker compose up
# Grafana: http://localhost:3000  (admin/admin)
# Prometheus: http://localhost:9090
```

---

## CI/CD ve Otomasyon Akışı

```
Her gün 07:00 (UTC 04:00)
        │
        ▼
GitHub Actions — daily_pipeline.yml
        │
        ├── pip install + HF model cache
        ├── python -m src.pipeline
        └── git commit data/analyzed/ + push

git push origin main
        │
        ├──► CI (ci.yml)
        │    ├── ruff check src/ tests/
        │    └── pytest tests/ -v
        │
        └──► Deploy (deploy.yml)
             └── curl RENDER_DEPLOY_HOOK_URL
                      │
                      ▼
                 Render.com
                 docker build + deploy

Yerel çalıştırma:
        │
        ▼
crontab: 0 7 * * * scripts/run_daily.sh
        │
        └── log: logs/pipeline_YYYY-MM-DD.log
```

### Cron Kurulumu (yerel)

```bash
# crontab'a ekle
crontab -e
# Şu satırı ekle:
0 7 * * * /Users/efeyol11/sementic_news/scripts/run_daily.sh

# Manuel test
bash scripts/run_daily.sh
tail -f logs/pipeline_$(date +%Y-%m-%d).log
```

### GitHub Actions Manuel Tetikleme

```
GitHub → Actions → Daily Pipeline → Run workflow
```

---

## Tech Stack Özeti

| Katman | Teknoloji | Neden |
|--------|-----------|-------|
| Veri toplama | feedparser | RSS/Atom parse, hata toleranslı |
| Metin temizleme | BeautifulSoup, langdetect | HTML strip, dil tespiti |
| Sentiment | HuggingFace Transformers | Türkçe BERT, hazır model |
| NER | HuggingFace Transformers | Türkçe NER, hazır model |
| Kümeleme | scikit-learn TF-IDF + KMeans | Hafif, yorumlanabilir |
| Experiment tracking | MLflow | Ücretsiz, yerel SQLite |
| Pipeline versiyonlama | DVC | Git ile entegre, cache |
| API | FastAPI | Hızlı, otomatik OpenAPI docs |
| Monitoring | Prometheus + Grafana | Endüstri standardı |
| CI/CD | GitHub Actions + Render | Ücretsiz tier |
| Loglama | loguru | Renkli, yapılandırılmış |
| Dashboard | Next.js 14 + Tailwind + Recharts | App Router, dark enterprise UI |

---

## Dizin Yapısı

```
sementic_news/
├── src/
│   ├── data/
│   │   ├── rss_collector.py    # Adım 1: RSS → data/raw/
│   │   ├── preprocessor.py     # Adım 2: raw → data/processed/
│   │   └── dataset.py          # HuggingFace dataset (eğitim için)
│   ├── analysis/
│   │   ├── sentiment.py        # Adım 3: processed → analyzed (sentiment)
│   │   ├── ner.py              # Adım 4: analyzed += entities
│   │   └── clustering.py       # Adım 5: analyzed += cluster_id
│   ├── training/
│   │   ├── train.py            # BERT fine-tune (isteğe bağlı)
│   │   ├── evaluate.py         # F1/accuracy hesaplama
│   │   └── registry.py         # MLflow model registry
│   ├── api/
│   │   └── main.py             # FastAPI uygulaması
│   └── pipeline.py             # Tüm adımları tek komutta çalıştırır
├── data/
│   ├── raw/                    # YYYY-MM-DD.json (ham)
│   ├── processed/              # YYYY-MM-DD.json (temizlenmiş)
│   └── analyzed/               # YYYY-MM-DD.json + _clusters.json
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── notebooks/
│   └── eda.ipynb               # Görselleştirme (5 grafik)
├── tests/
│   └── test_api.py             # 6 endpoint smoke testi
├── .github/workflows/
│   ├── ci.yml                  # Lint + test
│   └── deploy.yml              # Render CD
├── dashboard/                  # Next.js 14 enterprise dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx        # Ana dashboard (istatistikler, sentiment, kümeler)
│   │   │   ├── sources/        # Kaynak × sentiment karşılaştırması
│   │   │   └── topic/[id]/     # Küme detay sayfası
│   │   ├── components/
│   │   │   ├── ui/             # StatCard, Navbar, EntityCloud, ClusterGrid...
│   │   │   └── charts/         # SentimentPieChart, SourceHeatmap
│   │   └── lib/                # api.ts (FastAPI client), utils.ts
│   └── package.json
├── dvc.yaml                    # DVC pipeline tanımı
├── docker-compose.yml          # API + Prometheus + Grafana
├── Dockerfile                  # Production container
└── pyproject.toml              # Bağımlılıklar + araç ayarları
```

---

## Next.js Dashboard

```
dashboard/
  src/app/
    page.tsx          → /        Ana dashboard
    sources/page.tsx  → /sources Kaynak analizi
    topic/[id]/       → /topic/3 Küme detayı
```

**Özellikler:**
- Glassmorphism kartlar + `#0A0F1E` dark background
- Animated counters (Framer Motion)
- Sentiment gauge (renk geçişli progress bar)
- Kaynak × sentiment heatmap (yatay stacked bar)
- Entity cloud (PER / ORG / LOC renk kodlu)
- Cluster grid → tıklanabilir, küme detayına yönlendirir

**Çalıştırma:**
```bash
cd dashboard
npm install
npm run dev          # http://localhost:3000
# API'nin de açık olması gerekiyor:
uvicorn src.api.main:app --port 8000
```

**Vercel Deploy:**
```bash
vercel --cwd dashboard   # dashboard/ klasörünü deploy eder
# NEXT_PUBLIC_API_URL=https://your-api.onrender.com
```

---

## Günlük Çalışma Senaryosu

```
Sabah 06:00 (cron — Blok D'de eklenecek)
        │
        ▼
python -m src.pipeline
        │
        ├── collect:    545 haber çekildi (10 kaynak)
        ├── preprocess: 520 kaldı (HTML temiz, Türkçe)
        ├── sentiment:  520 haber → positive/negative (2 dk)
        ├── ner:        520 haber → PER/ORG/LOC (5 dk)
        └── clustering: 15 konu kümesi oluşturuldu
                │
                ▼
        data/analyzed/2026-04-20.json  ✓
        data/analyzed/2026-04-20_clusters.json  ✓
                │
                ▼
        API sorgulanabilir:
        GET /api/today → günlük özet
        GET /api/topic/3 → ekonomi haberleri
        GET /api/source-comparison → Cumhuriyet %61 negatif
```
