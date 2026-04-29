"""Export the production sentiment model to ONNX with int8 quantization.

Run once per model release. The ONNX runtime path is ~3–5× faster than
PyTorch on CPU for our BERT-base sequence classifier, and dynamic int8
quantization shrinks the file size ~75% with negligible accuracy drop.

Outputs land under ``models/onnx/`` (fp32) and ``models/onnx_int8/``
(quantized). Both directories are gitignored — re-run this script after
each weekly retrain.

Usage:
    pip install -e ".[serving]"
    python -m src.serving.onnx_export
    python -m src.serving.onnx_export --model-id efeyol11/bert-turkish-sentiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "efeyol11/bert-turkish-sentiment"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ONNX_DIR = _REPO_ROOT / "models" / "onnx"
_ONNX_INT8_DIR = _REPO_ROOT / "models" / "onnx_int8"


def export(model_id: str = DEFAULT_MODEL_ID) -> tuple[Path, Path]:
    """Export *model_id* to ONNX (fp32) then derive a dynamic int8 build.

    Returns (fp32_dir, int8_dir).
    """
    logger.info(f"Exporting {model_id} → ONNX (fp32)...")
    _ONNX_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(str(_ONNX_DIR))
    tokenizer.save_pretrained(str(_ONNX_DIR))
    logger.success(f"FP32 ONNX saved → {_ONNX_DIR.relative_to(_REPO_ROOT)}")

    logger.info("Applying dynamic int8 quantization...")
    _ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(str(_ONNX_DIR))
    # ``avx2`` is the broadest CPU profile (works on most modern x86 hosts
    # incl. GitHub free runners and Render free tier). Switch to ``avx512``
    # or ``arm64`` later if the deploy target supports it.
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(_ONNX_INT8_DIR), quantization_config=qconfig)
    tokenizer.save_pretrained(str(_ONNX_INT8_DIR))

    # ``ORTQuantizer`` saves the artifact as ``model_quantized.onnx`` which
    # forces every loader to pass an explicit ``file_name`` arg. Rename to
    # the canonical ``model.onnx`` so the int8 dir is a drop-in replacement
    # for the fp32 dir and ``from_pretrained`` auto-discovers it.
    quantized_file = _ONNX_INT8_DIR / "model_quantized.onnx"
    canonical_file = _ONNX_INT8_DIR / "model.onnx"
    if quantized_file.exists():
        quantized_file.rename(canonical_file)

    logger.success(f"INT8 ONNX saved → {_ONNX_INT8_DIR.relative_to(_REPO_ROOT)}")

    return _ONNX_DIR, _ONNX_INT8_DIR


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the sentiment model to ONNX + int8")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    export(args.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
