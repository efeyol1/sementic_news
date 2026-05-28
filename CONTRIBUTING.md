# Contributing to Semantic News

Bu proje V2 multi-country geçişin ortasında; katkılar memnuniyetle, ama aşağıdaki workflow'a uyman gerekiyor.

## Geliştirme Ortamı

```bash
python3.10+ -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,pipeline]"

cp .env.example .env  # DATABASE_URL'i Neon connection string ile doldur
python -m src.db.schema
alembic upgrade head
```

## Branch / PR Workflow

- Branch naming: `sprint-N` (V2 phases için), `entity-sprint-N` (entity-narrative track için), `fix/<short-slug>` (bug fix), `docs/<short-slug>` (sadece doc).
- Her PR **tek concern** olmalı — feature ve refactor karıştırma.
- V2 phase kuralı: her sprint **backward-compatible** olmalı. Mevcut country YAML schema'sını kırma; yeni alan eklerken loader'da default ver.
- Country-specific bilgi **asla** Python koduna hardcoded olamaz. Her şey `configs/countries/*.yaml`'a gider. Bu mimari ilkeyi bozan PR reject edilir.
- Commit message: `<type>(<scope>): <summary>`. Tipler: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`. Örn: `feat(analysis): collocations module + stopwords config`.

## Testler

```bash
pytest tests/                          # tüm testler (unit + behavioral)
pytest -m "not behavioral"             # hızlı (~saniyeler)
pytest tests/behavioral/               # sentiment behavioral suite
ruff check src/ tests/                 # lint
```

**Behavioral gate uyarısı:** `tests/behavioral/test_sentiment_checklist.py` `must-pass` capability'leri retraining pipeline'ı bloke eder. Behavioral test fail eden bir model HF Hub'a push EDİLMEZ. Yeni capability eklerken `watchlist` veya `aspirational` xfail marker'larıyla başla, ölçüm yap, sonra `must-pass`'e terfi ettir.

## Lokal Pipeline Çalıştırma

```bash
# Bir ülke için tam zincir
python -m src.pipeline --country turkey

# Sadece RSS toplama (model yüklemeden)
python -m src.data.rss_collector --country germany

# Body fetch ile (opt-in, yavaş)
python -m src.pipeline --country turkey --fetch-articles --article-limit 100
```

## Yeni Ülke Ekleme

1. `configs/countries/<slug>.yaml` oluştur (örnek: `germany.yaml`). RSS/sitemap kaynaklarını listele, `license: "unknown"` alanını her source için doldur (ya da `robots-allowed`/`restricted`/`scraped-public` öğrenebildiğin kadar).
2. `entity_narrative.wikidata.enabled: false` ile başla; Wikidata cache hazırlandıktan sonra aç.
3. `python -m src.pipeline --country <slug> --date YYYY-MM-DD` ile lokal smoke test.
4. PR'da `data/qa/sentiment_review_<country>_*.csv` gold seed örneği bulundur (en az 200 örnek).

## Code Style

- Ruff (line-length=120, `target-version = "py310"`).
- Type hint'ler önerilir, zorunlu değil (mypy gate yok henüz).
- Docstring sadece "WHY" non-obvious ise; "WHAT" için iyi isimlendirme yeterli.

## Hangi Konular Tartışılır?

- ✅ Yeni ülke ekleme, RSS source genişletme
- ✅ Cluster/embedding iyileştirmesi (`project_clustering_plan.md` referans)
- ✅ Entity resolution / Wikidata cache iyileştirmesi
- ✅ Behavioral test capability genişletme
- ⚠ Framing analizi (Phase 10) — etik kurallarına dikkat: observable pattern'ler, "country X is Y" ifadeleri değil
- ❌ Tek başına sentiment skoru ile kişi/ülke/piyasa karar verme feature'ları (model card'da out-of-scope)

## İletişim

Soru veya öneri için issue aç. Güvenlik açığı için `SECURITY.md`'ye bak.
