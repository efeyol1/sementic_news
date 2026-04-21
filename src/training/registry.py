"""MLflow Model Registry helper.

En iyi training run'ını `turkish-news-sentiment` adıyla Model Registry'ye kaydeder
ve isteğe bağlı olarak Staging veya Production'a yükseltir.

Kullanım:
    python -m src.training.registry --list
    python -m src.training.registry --register
    python -m src.training.registry --register --promote staging
    python -m src.training.registry --promote production --version 2
"""

import argparse
import sys
from pathlib import Path

import mlflow
from loguru import logger
from mlflow import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = f"sqlite:///{_REPO_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "turkish-sentiment"
REGISTERED_MODEL_NAME = "turkish-news-sentiment"
METRIC = "eval_f1_macro"          # best run bu metriğe göre seçilir
PROMOTE_THRESHOLD = 0.75          # bu altındaki F1 Production'a çıkmaz

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_best_run() -> mlflow.entities.Run:
    """Experiment içindeki en yüksek METRIC değerine sahip run'ı döndür."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' bulunamadı. "
            "Önce `python -m src.training.train` çalıştırın."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{METRIC} > 0",
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"'{EXPERIMENT_NAME}' içinde {METRIC} metriği olan run bulunamadı."
        )
    return runs[0]


def _ensure_registered_model() -> None:
    """Registry'de model yoksa oluştur."""
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description="Turkish news sentiment classifier (BERT fine-tuned)",
        )
        logger.info(f"Registered model oluşturuldu: {REGISTERED_MODEL_NAME!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_versions() -> None:
    """Registry'deki tüm model versiyonlarını listele."""
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except mlflow.exceptions.MlflowException:
        logger.warning(f"'{REGISTERED_MODEL_NAME}' henüz registry'de yok.")
        return

    if not versions:
        logger.info("Kayıtlı versiyon yok.")
        return

    logger.info(f"\n{'Ver':>4}  {'Stage':<12}  {'Run ID':>12}  {METRIC}")
    logger.info("-" * 55)
    for v in sorted(versions, key=lambda x: int(x.version)):
        run = client.get_run(v.run_id)
        f1 = run.data.metrics.get(METRIC, 0.0)
        aliases = ", ".join(v.aliases) if v.aliases else "-"
        logger.info(
            f"{v.version:>4}  {aliases:<12}  {v.run_id[:8]}...  {f1:.4f}"
        )


def register_best() -> str:
    """En iyi run'ı registry'ye kaydet, yeni versiyon numarasını döndür."""
    run = _get_best_run()
    f1 = run.data.metrics.get(METRIC, 0.0)
    logger.info(
        f"En iyi run: {run.info.run_id[:8]}... | {METRIC}={f1:.4f}"
    )

    _ensure_registered_model()

    model_uri = f"runs:/{run.info.run_id}/best"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    logger.success(
        f"Model kayıt edildi → {REGISTERED_MODEL_NAME} v{mv.version} "
        f"(run: {run.info.run_id[:8]}...)"
    )
    return mv.version


def push_to_hub(model_path: str, hub_model_id: str) -> None:
    """Push a trained model from *model_path* to HuggingFace Hub."""
    logger.info(f"Pushing {model_path} → {hub_model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
    logger.success(f"Model pushed to HuggingFace Hub: {hub_model_id}")


def get_best_f1() -> float:
    """Return the highest eval_f1_macro across all runs in the experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return 0.0
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{METRIC} > 0",
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        return 0.0
    return runs[0].data.metrics.get(METRIC, 0.0)


def promote(version: str, stage: str) -> None:
    """Bir model versiyonunu 'staging' veya 'production' alias'ına yükselt.

    Args:
        version: Yükseltilecek versiyon numarası (str).
        stage: 'staging' veya 'production'.
    """
    stage = stage.lower()
    if stage not in ("staging", "production"):
        raise ValueError("stage 'staging' veya 'production' olmalı.")

    run_id = client.get_model_version(REGISTERED_MODEL_NAME, version).run_id
    f1 = client.get_run(run_id).data.metrics.get(METRIC, 0.0)

    if stage == "production" and f1 < PROMOTE_THRESHOLD:
        raise ValueError(
            f"v{version} F1={f1:.4f} < eşik {PROMOTE_THRESHOLD} — "
            "Production'a yükseltme reddedildi."
        )

    # Önce aynı alias'ı başka versiyondan kaldır
    try:
        existing = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, stage)
        if existing.version != version:
            client.delete_registered_model_alias(REGISTERED_MODEL_NAME, stage)
            logger.debug(f"Eski {stage} alias'ı v{existing.version}'dan kaldırıldı.")
    except mlflow.exceptions.MlflowException:
        pass

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, stage, version)
    logger.success(
        f"v{version} → {stage.upper()} ({METRIC}={f1:.4f})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry yönetim aracı."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Kayıtlı model versiyonlarını listele",
    )
    parser.add_argument(
        "--register", action="store_true",
        help="En iyi run'ı registry'ye kaydet",
    )
    parser.add_argument(
        "--promote", choices=["staging", "production"],
        metavar="STAGE",
        help="Versiyonu 'staging' veya 'production'a yükselt",
    )
    parser.add_argument(
        "--version",
        help=(
            "Yükseltilecek versiyon "
            "(--promote ile kullanılır, default: son kaydedilen)"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if not any([args.list, args.register, args.promote]):
        logger.error("En az bir flag gerekli: --list, --register, --promote")
        sys.exit(1)

    try:
        if args.list:
            list_versions()

        registered_version = None
        if args.register:
            registered_version = register_best()

        if args.promote:
            version = args.version or registered_version
            if version is None:
                logger.error(
                    "--promote için --version belirtin "
                    "veya --register ile birlikte kullanın."
                )
                sys.exit(1)
            promote(version, args.promote)

    except (RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
