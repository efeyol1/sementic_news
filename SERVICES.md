# Servisler ve Çalıştırma Rehberi

## Projenin Çalışan Servisleri

---

### 1. FastAPI (`uvicorn src.api.main:app --port 8000`)
**Ne yapar:** ML pipeline'ının çıktısını (`data/analyzed/`) okuyup HTTP endpoint'lere dönüştürür. Dashboard'un veri kaynağı.

```
GET /api/today          → günlük özet (sentiment, entity, cluster)
GET /api/topic/3        → küme detayı + haberler
GET /api/source-comparison → kaynak bazlı karşılaştırma
GET /metrics            → Prometheus için ham metrikler
```

**Nasıl çalıştırılır:**
```bash
cd sementic_news
source .venv/bin/activate
uvicorn src.api.main:app --port 8000
```

---

### 2. Next.js Dashboard (`npm run dev`)
**Ne yapar:** FastAPI'ye istek atıp verileri görselleştirir. Kullanıcıya sunulan arayüz.

```
/           → günlük analiz (stat kartlar, sentiment, kümeler)
/sources    → kaynak × sentiment heatmap
/topic/[id] → tek bir kümenin detayı
```

**Nasıl çalıştırılır:**
```bash
cd dashboard
npm run dev   # → http://localhost:3000
```

---

### 3. Docker Compose (`docker compose up`)
**Ne yapar:** 3 servisi tek komutla ayağa kaldırır: FastAPI + Prometheus + Grafana. Production senaryosu.

```
docker-compose.yml
├── api         → FastAPI container (:8000)
├── prometheus  → Metrikleri toplar (:9090)
└── grafana     → Metrikleri görselleştirir (:3000)
```

**Nasıl çalıştırılır:**
```bash
cd sementic_news
docker compose up
```

> Docker varken ayrıca `uvicorn` çalıştırmana gerek yok — API zaten container içinde çalışıyor.

---

### 4. Prometheus (`docker compose up` ile gelir, :9090)
**Ne yapar:** FastAPI'nin `/metrics` endpoint'ini her 15 saniyede bir okur ve zaman serisi olarak saklar. Grafana'nın veri kaynağı.

```
FastAPI /metrics  →  Prometheus  →  Grafana
(ham sayaçlar)      (saklama)      (görselleştirme)
```

Doğrudan `http://localhost:9090` adresinden sorgu yazabilirsin ama genelde sadece Grafana üzerinden kullanılır.

---

### 5. Grafana (`docker compose up` ile gelir, :3000)
**Ne yapar:** API'nin teknik sağlığını izler — kaç istek geldi, yanıt süresi ne kadar, hata var mı. Dashboard ile karıştırma: o kullanıcı içindi, bu DevOps içindir.

```
http://localhost:3000
admin / admin

Paneller:
├── İstek/dakika
├── Ortalama yanıt süresi
├── HTTP 5xx hata oranı
└── Endpoint dağılımı
```

---

### Hangi durumda neyi açarsın?

| Senaryo | Komutlar |
|---|---|
| Sadece dashboard geliştirme | `uvicorn` + `npm run dev` |
| Production simülasyonu | `docker compose up` + `npm run dev` |
| Sadece API test | `uvicorn` + `curl` |
| Monitoring izleme | `docker compose up` → Grafana |

---

### Genel Akış

```
data/analyzed/*.json
        ↓
    FastAPI (:8000)
    ↙           ↘
Next.js        Prometheus
(:3000)          (:9090)
dashboard         ↓
               Grafana (:3000)
```
