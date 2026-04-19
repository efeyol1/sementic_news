# Semantic News TR

Türkiye'nin 10 büyük haber kaynağından RSS ile gerçek zamanlı veri toplayıp semantik meta-analiz yapan açık kaynak platform.

## Ne Yapar?

- 10 Türk haber kaynağından RSS feed çeker (Habertürk, Hürriyet, NTV, CNN Türk, Sözcü, Milliyet, Sabah, TRT Haber, Cumhuriyet, Yeni Şafak)
- Toplanan haberleri temizler, normalleştirir ve dil tespiti yapar
- BERT tabanlı Türkçe duygu analizi modeli ile haberleri sınıflandırır (pozitif / negatif / nötr)
- REST API üzerinden tahmin servisi sunar
- Vektör veritabanı ile semantik benzer haber arama yapar

## Mevcut Durum

**Phase 1 — ML Pipeline (devam ediyor)**

| Adım | Durum |
|------|-------|
| Proje iskeleti (DVC, MLflow) | ✅ Tamamlandı |
| Training pipeline (BERT fine-tune) | ✅ Tamamlandı |
| RSS collector (10/10 kaynak) | ✅ Tamamlandı |
| Preprocessor (temizleme, dil tespiti) | ✅ Tamamlandı |
| Full training run + model değerlendirme | ⬜ Devam ediyor |
| **Phase 2** — FastAPI servisi | ⬜ Planlandı |
| **Phase 3** — CI/CD (GitHub Actions) | ⬜ Planlandı |
| **Phase 4** — Monitoring + Vector DB | ⬜ Planlandı |

## Kurulum

```bash
git clone https://github.com/<kullanici>/semantic-news-tr.git
cd semantic-news-tr

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Kullanım

### Veri Toplama

```bash
# 10 kaynaktan RSS çek → data/raw/YYYY-MM-DD.json
python -m src.data.rss_collector
```

### Ön İşleme

```bash
# Bugünün verisini işle → data/processed/YYYY-MM-DD.json
python -m src.data.preprocessor

# Belirli bir tarih için
python -m src.data.preprocessor --date 2026-04-17
```

### Model Eğitimi (dry-run)

```bash
# 5 adımlık smoke test
python -m src.training.train --dry-run
```

## Proje Yapısı

```
semantic-news-tr/
├── src/
│   ├── data/
│   │   ├── rss_collector.py   # RSS veri toplama
│   │   ├── preprocessor.py    # Metin temizleme ve normalizasyon
│   │   └── dataset.py         # HuggingFace dataset yükleyici
│   ├── training/
│   │   ├── train.py           # BERT fine-tuning
│   │   └── evaluate.py        # Model değerlendirme
│   ├── api/                   # FastAPI servisi (Phase 2)
│   └── vector_db/             # Semantik arama (Phase 4)
├── data/
│   ├── raw/                   # Ham RSS verisi (DVC ile takip edilir)
│   └── processed/             # İşlenmiş veri (DVC ile takip edilir)
├── models/                    # Eğitilmiş model ağırlıkları
├── tests/                     # pytest test suite
└── pyproject.toml
```

## Telif ve Veri Notu

**Bu repo yalnızca kaynak kodu içerir, haber verisi içermez.**
Kullanıcı kendi kurulumunda `rss_collector.py` ile RSS üzerinden veri toplar.
Toplanan haberlerin telif hakkı ilgili yayın kuruluşlarına aittir.

## Lisans

[MIT](LICENSE)
