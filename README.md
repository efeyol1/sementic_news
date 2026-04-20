# Semantic News TR

Türkiye'nin 10 büyük haber kaynağından RSS ile günlük veri toplayıp sentiment analizi, NER ve konu kümeleme yapan tam MLOps projesi.

## Mevcut Durum

| Blok | İçerik | Durum |
|------|--------|-------|
| **A — ML Pipeline** | RSS collector, preprocessor, sentiment, NER, clustering | ✅ |
| **B — MLOps** | pipeline.py, DVC, model registry (MLflow) | ✅ |
| **C — DevOps** | FastAPI, Dockerfile, GitHub Actions CI/CD | ✅ |
| **C — Deploy** | Render deploy, Grafana monitoring | 🔄 |
| **D — Birleştirme** | Cron job, React dashboard, fine-tune | ⬜ |

## Özellikler

- 10 Türk haber kaynağından RSS çeker (Habertürk, Hürriyet, NTV, CNN Türk, Sözcü, Milliyet, Sabah, TRT Haber, Cumhuriyet, Yeni Şafak)
- BERT tabanlı Türkçe sentiment analizi (`savasy/bert-base-turkish-sentiment-cased`)
- Named Entity Recognition — PER / ORG / LOC (`savasy/bert-base-turkish-ner-cased`)
- TF-IDF + KMeans ile 15 konu kümesi
- MLflow experiment tracking + model registry
- DVC ile pipeline versiyonlama
- FastAPI REST servisi
- GitHub Actions CI (lint + test) + Render CD

## Kurulum

```bash
git clone https://github.com/<kullanici>/semantic-news-tr.git
cd semantic-news-tr

python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## Pipeline Kullanımı

```bash
# Tek komutla tam zincir (bugün)
python -m src.pipeline

# Belirli tarih
python -m src.pipeline --date 2026-04-20

# RSS zaten çekildiyse
python -m src.pipeline --skip-collect

# Adım adım
python -m src.data.rss_collector
python -m src.data.preprocessor
python -m src.analysis.sentiment
python -m src.analysis.ner
python -m src.analysis.clustering

# DVC ile
dvc repro
```

## Model Registry

```bash
python -m src.training.train --dry-run        # smoke test
python -m src.training.registry --register    # en iyi run'ı kaydet
python -m src.training.registry --register --promote staging
python -m src.training.registry --list
```

## API

```bash
uvicorn src.api.main:app --reload
```

| Endpoint | Açıklama |
|----------|----------|
| `GET /health` | Liveness probe |
| `GET /api/today` | Günlük analiz özeti |
| `GET /api/topic/{id}` | Cluster haberleri |
| `GET /api/source-comparison` | Kaynak bazlı sentiment |
| `GET /metrics` | Prometheus metrikleri |

## Testler

```bash
pytest tests/ -v
```

## Proje Yapısı

```
sementic_news/
├── src/
│   ├── data/
│   │   ├── rss_collector.py    # RSS veri toplama
│   │   ├── preprocessor.py     # Temizleme, dil tespiti
│   │   └── dataset.py          # HuggingFace dataset
│   ├── analysis/
│   │   ├── sentiment.py        # Sentiment analizi
│   │   ├── ner.py              # Named entity recognition
│   │   └── clustering.py       # Konu kümeleme
│   ├── training/
│   │   ├── train.py            # BERT fine-tuning
│   │   ├── evaluate.py         # Metrik hesaplama
│   │   └── registry.py         # MLflow model registry
│   ├── api/
│   │   └── main.py             # FastAPI uygulaması
│   └── pipeline.py             # Tam zincir orkestratörü
├── data/
│   ├── raw/                    # Ham RSS verisi (DVC)
│   ├── processed/              # İşlenmiş veri (DVC)
│   └── analyzed/               # Analiz sonuçları (DVC)
├── notebooks/
│   └── eda.ipynb               # Keşifsel veri analizi
├── tests/
│   └── test_api.py             # API smoke testleri
├── .github/workflows/
│   ├── ci.yml                  # Lint + test
│   └── deploy.yml              # Render deploy
├── dvc.yaml                    # DVC pipeline
├── Dockerfile
└── pyproject.toml
```

## Telif ve Veri Notu

**Bu repo yalnızca kaynak kodu içerir, haber verisi içermez.**
Kullanıcı kendi kurulumunda `rss_collector.py` ile RSS üzerinden veri toplar.
Toplanan haberlerin telif hakkı ilgili yayın kuruluşlarına aittir.

## Lisans

[MIT](LICENSE)
