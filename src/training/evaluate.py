"""Evaluation utilities for the Turkish sentiment model.

Exposes a single ``compute_metrics`` callable used by the HuggingFace
``Trainer`` plus a couple of helpers for richer per-class reporting.
The Trainer already flattens the dict returned here into MLflow metrics
(prefixed with ``eval_``), so adding per-class keys is enough to land them
in MLflow without any extra logging code at the call site.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import EvalPrediction

from src.data.dataset import ID2LABEL


def _per_class_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Per-class precision/recall/F1 keyed by ``ID2LABEL`` names.

    Only labels that exist in ``ID2LABEL`` are reported; missing classes
    in the eval batch are reported as ``0.0`` so the keys are stable
    across runs.
    """
    label_ids = sorted(ID2LABEL.keys())
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    out: dict[str, float] = {}
    for idx, label_id in enumerate(label_ids):
        name = ID2LABEL[label_id]
        out[f"precision_{name}"] = round(float(precisions[idx]), 4)
        out[f"recall_{name}"] = round(float(recalls[idx]), 4)
        out[f"f1_{name}"] = round(float(f1s[idx]), 4)
    return out


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """HuggingFace Trainer ``compute_metrics`` callback.

    Returns a flat dict so the Trainer can log each value as its own
    MLflow metric (``eval_f1_macro``, ``eval_f1_negative``, ...).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    accuracy = (preds == labels).mean()

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }
    metrics.update(_per_class_metrics(np.asarray(labels), np.asarray(preds)))
    return metrics


def print_classification_report(labels, preds, id2label: dict) -> None:
    label_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(labels, preds, target_names=label_names))


def confusion_matrix_dict(labels, preds, id2label: dict) -> dict:
    """JSON-friendly confusion matrix for ``mlflow.log_dict``.

    Shape: rows = true label, columns = predicted label, both ordered by
    label id. Returned dict is human-readable and survives a round trip
    through MLflow's artifact store without needing matplotlib.
    """
    label_ids = sorted(id2label.keys())
    label_names = [id2label[i] for i in label_ids]
    cm = confusion_matrix(labels, preds, labels=label_ids)
    return {
        "labels": label_names,
        "matrix": cm.tolist(),
        "row_axis": "true_label",
        "col_axis": "predicted_label",
    }


def min_per_class_f1(metrics: dict) -> float:
    """Smallest ``f1_<class>`` across all classes from a metrics dict.

    Accepts either bare keys (``f1_negative``) or Trainer-prefixed keys
    (``eval_f1_negative``). Missing classes are treated as 0.0.
    """
    f1s: list[float] = []
    for label_id, name in ID2LABEL.items():
        for key in (f"f1_{name}", f"eval_f1_{name}"):
            if key in metrics:
                f1s.append(float(metrics[key]))
                break
        else:
            f1s.append(0.0)
    return min(f1s) if f1s else 0.0
