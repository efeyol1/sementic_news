"""Benchmark PyTorch vs ONNX-fp32 vs ONNX-int8 sentiment inference.

Reports per-backend p50 / p95 / p99 latency, throughput, and prediction
agreement (vs the PyTorch baseline). Also prints a Markdown table you can
paste straight into the README.

Acts as a CI gate: exits non-zero if the int8 backend's prediction
agreement falls below ``--min-agreement`` (default 0.99) or if either
ONNX backend isn't faster than PyTorch.

Usage:
    pip install -e ".[serving]"
    python -m src.serving.onnx_export       # populates models/onnx*/
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --repeats 10 --min-agreement 0.98
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ``optimum`` is in the ``serving`` extra. Lazy-import it inside the ONNX
# predictor so the module remains importable (and unit-testable) without
# the extra installed.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ONNX_DIR = _REPO_ROOT / "models" / "onnx"
_ONNX_INT8_DIR = _REPO_ROOT / "models" / "onnx_int8"

DEFAULT_HF_ID = "efeyol11/bert-turkish-sentiment"

# Production-shaped Turkish news headlines. Mix of polarity and length so
# the latency distribution is representative; agreement is computed across
# all of them so a regression on any class shows up.
SEED_TEXTS: list[str] = [
    "Ekonomi büyüdü ve enflasyon geriledi.",
    "Borsa kapanışta yatay seyretti.",
    "Milli takım maçı 3-0 kazandı, taraftarlar mutlu.",
    "Şirket büyük zarar açıkladı, hisseler düştü.",
    "Bakanlık yarın bir basın toplantısı düzenleyecek.",
    "Deprem nedeniyle çok sayıda bina yıkıldı.",
    "İhracat geçen yılın aynı dönemine göre rekor kırdı.",
    "Saldırıda çok sayıda kişi hayatını kaybetti.",
    "Yeni hastane hizmete açıldı, halk teşekkür etti.",
    "Anlaşma çöktü, görüşmeler askıya alındı.",
    "Cumhurbaşkanı bugün Ankara'da bir programa katılacak.",
    "Hava sıcaklığı mevsim normalleri civarında seyredecek.",
    "Bilim insanları kanser tedavisinde önemli bir ilerleme kaydetti.",
    "Yangın söndürme çalışmaları devam ediyor.",
    "Yeni yönetmelik resmi gazetede yayımlandı.",
    "Şirket çeyrek karını ikiye katladı.",
    "Hastalar tedaviye yanıt vermedi.",
    "Öğrenci uluslararası yarışmada altın madalya kazandı.",
    "Toplantı saat 14.00'te başlayacak.",
    "Kriz büyüdü, fiyatlar kontrolden çıktı.",
]


# ---------------------------------------------------------------------------
# Predictor wrappers — same signature for every backend
# ---------------------------------------------------------------------------


def _pytorch_predictor(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    id2label = model.config.id2label

    @torch.no_grad()
    def predict(text: str) -> str:
        inp = tok(text, return_tensors="pt", truncation=True, max_length=128)
        out = model(**inp).logits
        return id2label[int(out.argmax(dim=-1).item())]

    return predict


def _onnx_predictor(directory: Path):
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} not found. Run `python -m src.serving.onnx_export` first."
        )
    from optimum.onnxruntime import ORTModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(str(directory))
    model = ORTModelForSequenceClassification.from_pretrained(str(directory))
    id2label = model.config.id2label

    def predict(text: str) -> str:
        inp = tok(text, return_tensors="pt", truncation=True, max_length=128)
        out = model(**inp).logits
        return id2label[int(out.argmax(dim=-1).item())]

    return predict


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _bench(name: str, predict_fn, texts: list[str], repeats: int) -> dict:
    # Warmup eats first-call import / kernel jit overhead so the timed
    # window measures steady-state inference, not first-load.
    for _ in range(3):
        predict_fn(texts[0])

    latencies_ms: list[float] = []
    preds: list[str] = []
    t_start = time.perf_counter()
    for _ in range(repeats):
        for text in texts:
            t0 = time.perf_counter()
            label = predict_fn(text)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            preds.append(label)
    total_s = time.perf_counter() - t_start

    return {
        "name": name,
        "p50_ms": round(statistics.median(latencies_ms), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "throughput_qps": round(len(latencies_ms) / total_s, 1),
        "preds": preds,
    }


def _agreement(reference: list[str], candidate: list[str]) -> float:
    matches = sum(1 for a, b in zip(reference, candidate, strict=True) if a == b)
    return matches / len(reference) if reference else 0.0


def _markdown_table(rows: list[dict], baseline_p95: float) -> str:
    headers = "| Backend | p50 (ms) | p95 (ms) | p99 (ms) | QPS | Speedup vs PyTorch | Agreement |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [headers, sep]
    for row in rows:
        speedup = baseline_p95 / row["p95_ms"] if row["p95_ms"] else 0.0
        agreement = (
            f"{row['agreement']:.1%}" if "agreement" in row else "_baseline_"
        )
        lines.append(
            f"| {row['name']} | {row['p50_ms']} | {row['p95_ms']} | "
            f"{row['p99_ms']} | {row['throughput_qps']} | {speedup:.2f}× | "
            f"{agreement} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark sentiment inference backends")
    p.add_argument("--model-id", default=DEFAULT_HF_ID)
    p.add_argument("--repeats", type=int, default=5, help="Loops over the seed corpus")
    p.add_argument(
        "--min-agreement",
        type=float,
        default=0.99,
        help="Fail if any ONNX backend's prediction agreement vs PyTorch falls below this",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger.info(
        f"Benchmarking {args.model_id} — {len(SEED_TEXTS)} seeds × {args.repeats} repeats"
    )

    # PyTorch baseline first so we can use its predictions as the agreement reference.
    logger.info("Loading PyTorch backend...")
    pt = _bench("PyTorch", _pytorch_predictor(args.model_id), SEED_TEXTS, args.repeats)

    logger.info("Loading ONNX (fp32) backend...")
    onnx_fp32 = _bench("ONNX fp32", _onnx_predictor(_ONNX_DIR), SEED_TEXTS, args.repeats)
    onnx_fp32["agreement"] = _agreement(pt["preds"], onnx_fp32["preds"])

    logger.info("Loading ONNX (int8) backend...")
    onnx_int8 = _bench("ONNX int8", _onnx_predictor(_ONNX_INT8_DIR), SEED_TEXTS, args.repeats)
    onnx_int8["agreement"] = _agreement(pt["preds"], onnx_int8["preds"])

    print("\n## Inference benchmark — Turkish sentiment (CPU)\n")
    print(_markdown_table([pt, onnx_fp32, onnx_int8], baseline_p95=pt["p95_ms"]))
    print(
        f"\n_{len(SEED_TEXTS) * args.repeats} predictions per backend, "
        f"BERT-base, max_length=128, CPU only._\n"
    )

    # Gate: agreement must hold and ONNX must actually be faster.
    failures: list[str] = []
    for row in (onnx_fp32, onnx_int8):
        if row["agreement"] < args.min_agreement:
            failures.append(
                f"{row['name']} agreement {row['agreement']:.3f} < {args.min_agreement}"
            )
        if row["p95_ms"] >= pt["p95_ms"]:
            failures.append(
                f"{row['name']} p95 {row['p95_ms']}ms ≥ PyTorch {pt['p95_ms']}ms"
            )

    if failures:
        for msg in failures:
            logger.error(msg)
        return 1

    logger.success("Benchmark passed all gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
