# Semantic News TR — Proje Bağlamı

## Ne yapıyor
Türk haber kaynaklarından (RSS) günlük haber toplar, sentiment analizi + NER + konu kümeleme yapar. Öğrenme amaçlı full MLOps projesi.

## Mimari kararlar
- Pre-trained HuggingFace modelleri (fine-tune ikinci katman)
- RSS-first (scraping yok) — telif güvenli
- Tam metin saklanmıyor, sadece metadata + analiz sonuçları
- Production maliyeti: sıfır (free tier stack)
- Kod açık kaynak, veri açık kaynak DEĞİL

## Tech Stack
- ML: HuggingFace Transformers, MLflow, DVC, scikit-learn
- API: FastAPI
- CI/CD: GitHub Actions → Render/Railway
- Monitoring: Grafana
- Dashboard: React (Vercel)

## Pipeline (günlük çalışma sırası)
```
python -m src.data.rss_collector       # data/raw/YYYY-MM-DD.json
python -m src.data.preprocessor        # data/processed/YYYY-MM-DD.json
python -m src.analysis.sentiment       # data/analyzed/YYYY-MM-DD.json (sentiment ekler)
python -m src.analysis.ner             # data/analyzed/YYYY-MM-DD.json (entities ekler)
python -m src.analysis.clustering      # data/analyzed/YYYY-MM-DD.json + _clusters.json
```

## Modeller
| Modül | Model | Görev |
|---|---|---|
| sentiment.py | savasy/bert-base-turkish-sentiment-cased | positive/negative |
| ner.py | savasy/bert-base-turkish-ner-cased | PER, ORG, LOC |
| clustering.py | TF-IDF + KMeans (sklearn) | 15 topic cluster |

## Veri şeması (data/analyzed/YYYY-MM-DD.json)
Her item şu alanları içerir:
- Preprocessor: `title, summary, source_name, published_date, link, category, cleaned_title, cleaned_summary, is_turkish, char_count`
- Sentiment: `sentiment_label, sentiment_score, sentiment_scores, analyzed_at`
- NER: `entities {PER, ORG, LOC}, entity_count`
- Clustering: `cluster_id, cluster_keywords`

## Tamamlanan bloklar

### Blok A — ML Pipeline ✅
- [x] src/analysis/sentiment.py
- [x] src/analysis/ner.py
- [x] src/analysis/clustering.py
- [x] notebooks/eda.ipynb (görselleştirme)

### Blok B — MLOps (sıradaki)
- [ ] src/pipeline.py — tek komutla tam zincir
- [ ] dvc.yaml — DVC pipeline tanımı
- [ ] Evaluation pipeline (baseline metrikler)
- [ ] Model registry

### Blok C — DevOps
- [ ] Dockerfile
- [ ] src/api/ — FastAPI (/api/today, /api/topic/{id}, /api/source-comparison)
- [ ] GitHub Actions CI/CD
- [ ] Render/Railway deploy
- [ ] Health check + loglama + Grafana

### Blok D — Birleştirme
- [ ] Cron job (günlük otomatik pipeline)
- [ ] React dashboard (Vercel)
- [ ] API dokümantasyonu
- [ ] Fine-tune deneyleri

## MLflow deneyleri
- `turkish-sentiment` — model eğitimi
- `news-sentiment` — günlük sentiment analizi
- `news-ner` — günlük NER
- `news-clustering` — günlük kümeleme

## RSS Kaynakları (10 Türk haber sitesi)
Habertürk, Hürriyet, NTV, CNN Türk, Sözcü, Milliyet, Sabah, TRT Haber, Cumhuriyet, Yeni Şafak

## Kod kuralları
- `loguru.logger` — tüm loglar
- `pathlib.Path` — dosya yolları
- `json.dump(ensure_ascii=False)` — Türkçe karakterler
- Private helpers: `_underscore_prefix`
- Her modül `__main__` entry point + `--date YYYY-MM-DD` argümanı içerir
