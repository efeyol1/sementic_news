# Semantic News TR — Mimari

## Büyük Resim

```
┌─────────────────────────────────────────────────────────────────┐
│                     GÜNLÜK PIPELINE (04:00 UTC)                 │
│                                                                 │
│  RSS Kaynakları (10)                                            │
│  ──────────────────►  rss_collector  ──► PostgreSQL (Neon)      │
│                              │                                  │
│                        preprocessor  (clean, langdetect)        │
│                              │                                  │
│                         sentiment    (xlm-roberta zero-shot)    │
│                              │                                  │
│                           ner         (bert-turkish-ner)        │
│                              │                                  │
│                        clustering    (MiniLM + KMeans)          │
│                              │                                  │
│                       vector_store   (MiniLM → pgvector)        │
└──────────────────────────────┼──────────────────────────────────┘
                               │ PostgreSQL (Neon)
                     ┌─────────▼──────────┐
                     │     FastAPI         │
                     │  (Render, :8000)    │
                     └─────────┬──────────┘
                               │ REST + pgvector similarity
                     ┌─────────▼──────────┐
                     │   Next.js Dashboard │
                     │   (Vercel, :3000)   │
                     └────────────────────┘
```

---

## Veri Katmanı

### PostgreSQL — `news_items` tablosu

Her haber bir satır. Pipeline adımları sırayla aynı satırı günceller.

| Kolon | Tip | Doldurulduğu Adım |
|-------|-----|-------------------|
| `collected_date` | DATE | rss_collector |
| `title, summary, source_name, link` | TEXT | rss_collector |
| `cleaned_title, cleaned_summary` | TEXT | preprocessor |
| `is_turkish, char_count` | BOOL/INT | preprocessor |
| `sentiment_label, sentiment_score` | TEXT/FLOAT | sentiment |
| `sentiment_scores` | JSONB | sentiment |
| `entities, entity_count` | JSONB/INT | ner |
| `cluster_id, cluster_keywords, cluster_title` | INT/JSONB/TEXT | clustering |
| `embedding` | vector(384) | vector_store |

### PostgreSQL — `cluster_summaries` tablosu

Her gün 15 küme özeti: `(date, cluster_id) UNIQUE`, keywords, sentiment dağılımı, kaynak dağılımı.

### pgvector

`embedding vector(384)` kolonu + `HNSW` indeksi. `/api/similar` için cosine similarity sorgusu:
```sql
SELECT title, 1 - (embedding <=> $1::vector) AS similarity
FROM news_items ORDER BY embedding <=> $1::vector LIMIT 5
```

---

## Pipeline Adımları

### 1. `rss_collector.py`
- 10 RSS feed → feedparser → normalize
- `INSERT ... ON CONFLICT (link) DO NOTHING` ile dedup
- **Çıktı:** ~400-500 satır/gün

### 2. `preprocessor.py`
- HTML strip, whitespace normalize, langdetect
- 10 karakterden kısa olanları DELETE
- **Çıktı:** ~450 satır (is_turkish=True ~%99)

### 3. `sentiment.py`
- Model: `joeddav/xlm-roberta-large-xnli` (zero-shot classification)
- Türkçe etiketler: "olumlu haber" / "olumsuz haber" / "tarafsız haber"
- Çok dilli: TR/DE/FR/ES/EN SENTIMENT_LABELS dict'i genişlet
- **Çıktı:** sentiment_label, sentiment_score (0-1), sentiment_scores {pos/neg/neu}

### 4. `ner.py`
- Model: `savasy/bert-base-turkish-ner-cased`
- Entity tipleri: PER, ORG, LOC
- **Çıktı:** ~1700 entity/gün

### 5. `clustering.py`
- `paraphrase-multilingual-MiniLM-L12-v2` ile embedding
- KMeans (n=15), silhouette score MLflow'a log
- TF-IDF + Türkçe stopword listesi ile küme başlıkları
- **Çıktı:** cluster_id (0-14) + cluster_summaries tablosu

### 6. `vector_store.py`
- Aynı MiniLM modeli (lazy singleton)
- `vector(384)` → PostgreSQL pgvector
- **Çıktı:** her Türkçe habere 384 boyutlu embedding

---

## API Katmanı

```
FastAPI (Render)
├── GET /health                    → liveness probe
├── GET /api/today?date=           → günlük özet (sentiment, entity, cluster)
├── GET /api/topic/{id}?date=      → küme detayı + haberler
├── GET /api/source-comparison     → kaynak bazlı sentiment istatistik
├── GET /api/similar?q=&n=         → pgvector cosine similarity arama
└── GET /metrics                   → Prometheus metrikleri
```

Startup'ta `init_db()` çağrılır → tablolar yoksa yaratır.

---

## Dashboard Katmanı

```
Next.js 14 App Router (Vercel)
├── /                    → Ana sayfa: istatistik, sentiment, entity cloud, kümeler
├── /topic/[id]          → Küme detayı: haberler, entity cloud, benzer haberler
└── /sources             → Kaynak karşılaştırma: heatmap, sentiment dağılımı
```

API'ye `fetch` ile bağlanır (`NEXT_PUBLIC_API_URL` env var). 5 dakika revalidate.

---

## CI/CD

```
git push main
    │
    ├── deploy.yml
    │   ├── ruff check + pytest (PostgreSQL service container ile)
    │   └── Render deploy hook → API yeniden deploy
    │
    └── Vercel (otomatik Next.js deploy)

Cron: daily_pipeline.yml → 04:00 UTC → python -m src.pipeline
Cron: weekly_retrain.yml → Pazar 02:00 UTC → python -m src.training.retrain
```

---

## Yol Haritası

### Kısa Vade (1-2 ay)

| # | Ne | Neden |
|---|-----|-------|
| 1 | **Trend grafikleri** — çok günlü sentiment çizgi grafiği | Tek günlük veri yetersiz, örüntü görünmez |
| 2 | **Date picker UI** — takvim bileşeni | URL ile tarih seçmek kullanıcı dostu değil |
| 3 | **Türkçe news corpus ile fine-tune** | Zero-shot iyi ama domain-specific model daha hassas |
| 4 | **Backfill scripti** — geçmiş günleri toplu işle | Tarihsel veri olmadan trend gösterilemez |

### Orta Vade (3-6 ay)

| # | Ne | Neden |
|---|-----|-------|
| 5 | **Medya bias skoru** — aynı olayı farklı kaynakların nasıl çerçevelediği | Projenin en özgün katkısı olabilir |
| 6 | **Entity zaman serisi** — politikacı/kurum sentiment trendi | "Son 30 günde Erdoğan haberleri nasıl değişti?" |
| 7 | **Real-time modu** — WebSocket ile anlık güncelleme | Breaking news için kritik |
| 8 | **Avrupa genişlemesi** — DE/FR/ES kaynakları ekle | SENTIMENT_LABELS altyapısı hazır, sadece kaynak ekle |

### Uzun Vade — Projenin Gidebileceği Yer

| Vizyon | Açıklama |
|--------|----------|
| **Medya Gözlemevi** | Hangi kaynak hangi konuları öne çıkarıyor, hangi olayları görmezden geliyor? Araştırmacı gazetecilik aracı. |
| **Haber API as a Service** | Geliştiricilere abonelikle günlük analiz verisi sat — medya şirketleri, akademisyenler, finans firmaları hedef kitle. |
| **LLM Özet Katmanı** | Her küme için GPT/Claude ile otomatik özet üret. "Bugün Türkiye'de ne oldu?" sorusuna tek paragrafta cevap. |
| **Alarm Sistemi** | Sentiment aniden negatife dönen konular için bildirim gönder. Kriz erken uyarı sistemi. |
| **Multi-modal** | Haber görsellerini de analiz et, başlık-görsel tutarsızlığını tespit et. |
| **Akademik Dataset** | Etiketli Türkçe haber sentiment dataseti yayınla — Türkçe NLP topluluğuna katkı. |

---

## Teknoloji Seçim Gerekçeleri

| Karar | Alternatif | Neden bu? |
|-------|-----------|-----------|
| Zero-shot (xlm-roberta) | Fine-tuned BERT | Eğitim verisi domain mismatch sorununu ortadan kaldırır |
| pgvector | ChromaDB | Ayrı servis yok, tek DB yeter; Neon ücretsiz destekliyor |
| Neon PostgreSQL | Supabase, Render PG | Serverless, generous free tier, pgvector built-in |
| Next.js App Router | CRA, Vite | Server components → API fetch sunucu tarafında, CORS yok |
| psycopg2 | SQLAlchemy ORM | Sorgular basit, ORM overkill; direkt SQL daha şeffaf |
