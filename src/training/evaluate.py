import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    accuracy = (preds == labels).mean()

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


def print_classification_report(labels, preds, id2label: dict) -> None:
    label_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(labels, preds, target_names=label_names))