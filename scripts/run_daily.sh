#!/bin/bash
# Günlük pipeline — crontab veya launchd ile çalıştırılır.
# Crontab örneği (her gün 07:00):
#   0 7 * * * /Users/efeyol11/sementic_news/scripts/run_daily.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "=== Pipeline başlıyor: $(date) ===" >> "$LOG_FILE"

source "$REPO/.venv/bin/activate"

python -m src.pipeline >> "$LOG_FILE" 2>&1

echo "=== Pipeline bitti: $(date) ===" >> "$LOG_FILE"
