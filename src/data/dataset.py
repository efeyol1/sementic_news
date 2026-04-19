from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from loguru import logger

MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# Internal label space (used by model + MLflow)
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Raw dataset labels → internal integer IDs
# Covers Turkish variants: notr/nötr, negatif, pozitif (with/without accent)
_RAW_LABEL_MAP: dict[str, int] = {
    "notr": 1,
    "nötr": 1,
    "neutral": 1,
    "negatif": 0,
    "negative": 0,
    "pozitif": 2,
    "positive": 2,
}


def _map_label(raw: str) -> int:
    key = raw.strip().lower()
    if key not in _RAW_LABEL_MAP:
        raise KeyError(
            f"Unknown label '{raw}'. Known labels: {list(_RAW_LABEL_MAP.keys())}"
        )
    return _RAW_LABEL_MAP[key]


def load_turkish_sentiment(split: str = "train") -> Dataset:
    logger.info(f"Loading dataset split={split}")
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset", split=split)
    return ds


def preprocess(batch: dict, tokenizer: AutoTokenizer) -> dict:
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = [_map_label(l) for l in batch["label"]]
    return tokens


def get_tokenized_datasets(tokenizer: AutoTokenizer, max_samples: int | None = None):
    train_ds = load_turkish_sentiment("train")
    val_ds = load_turkish_sentiment("test")

    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(max_samples // 4, len(val_ds))))
        logger.info(f"SMOKE TEST — using {len(train_ds)} train / {len(val_ds)} val samples")

    logger.info(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    train_ds = train_ds.map(
        lambda b: preprocess(b, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda b: preprocess(b, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    return train_ds, val_ds