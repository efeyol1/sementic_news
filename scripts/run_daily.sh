#!/bin/bash
# Günlük pipeline — crontab veya launchd ile çalıştırılır.
# Crontab örneği (her gün 07:00 ve 19:00):
#   0 7,19 * * * /Users/efeyol11/sementic_news/scripts/run_daily.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "=== Pipeline başlıyor: $(date) ===" >> "$LOG_FILE"

source "$REPO/.venv/bin/activate"

optional_failures=0

run_country() {
  local country="$1"
  local status=0
  echo "--- ${country} başlıyor: $(date) ---" >> "$LOG_FILE"

  if [ "$country" = "turkey" ]; then
    if SENTIMENT_MODEL_ID="efeyol11/bert-turkish-sentiment" \
      SENTIMENT_BACKEND="onnx_int8" \
      python -m src.pipeline --country "$country" >> "$LOG_FILE" 2>&1; then
      status=0
    else
      status=$?
    fi
  else
    unset SENTIMENT_MODEL_ID
    unset SENTIMENT_BACKEND
    if python -m src.pipeline --country "$country" >> "$LOG_FILE" 2>&1; then
      status=0
    else
      status=$?
    fi
  fi

  if [ "$status" -ne 0 ]; then
    echo "--- ${country} hata ile bitti: $(date), exit=${status} ---" >> "$LOG_FILE"
    if [ "$country" = "turkey" ]; then
      return "$status"
    fi
    optional_failures=$((optional_failures + 1))
    return 0
  fi

  echo "--- ${country} bitti: $(date) ---" >> "$LOG_FILE"
}

for country in turkey germany italy spain uk france; do
  run_country "$country"
done

if [ "$optional_failures" -gt 0 ]; then
  echo "=== Pipeline optional ülke hatalarıyla tamamlandı: ${optional_failures} ===" >> "$LOG_FILE"
fi

echo "=== Pipeline bitti: $(date) ===" >> "$LOG_FILE"
