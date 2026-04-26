from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoTokenizer

MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

_RAW_LABEL_MAP: dict[str, int] = {
    "negatif": 0,
    "negative": 0,
    "notr": 1,
    "nötr": 1,
    "neutral": 1,
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
    ds = load_dataset(
        "winvoker/turkish-sentiment-analysis-dataset", split=split
    )
    return ds


def preprocess(batch: dict, tokenizer: AutoTokenizer) -> dict:
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = [_map_label(lbl) for lbl in batch["label"]]
    return tokens


def get_tokenized_datasets(
    tokenizer: AutoTokenizer,
    max_samples: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
    shuffle_seed: int = 42,
):
    """Load and tokenize the dataset.

    Args:
        max_samples: smoke-test cap (mutually exclusive with start_idx/end_idx).
        start_idx, end_idx: slice the train set after a deterministic shuffle.
            Use these for incremental/chunked training.
        shuffle_seed: shuffle seed (kept constant across chunks so chunks
            don't overlap and together cover the full dataset).
    """
    train_ds = load_turkish_sentiment("train")
    val_ds = load_turkish_sentiment("test")

    train_ds = train_ds.shuffle(seed=shuffle_seed)

    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
        val_ds = val_ds.select(range(min(max_samples // 4, len(val_ds))))
        logger.info(f"SMOKE TEST — {len(train_ds)} train / {len(val_ds)} val samples")
    elif start_idx is not None or end_idx is not None:
        s = start_idx or 0
        e = min(end_idx or len(train_ds), len(train_ds))
        train_ds = train_ds.select(range(s, e))
        logger.info(f"CHUNK [{s}:{e}] — {len(train_ds)} train samples")

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
