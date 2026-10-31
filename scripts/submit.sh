#!/usr/bin/env bash
# Submit an experiment's predictions to Kaggle.
# Usage: bash scripts/submit.sh lgb_baseline "message"
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?experiment name, e.g. lgb_baseline}
MSG=${2:-$NAME}
kaggle competitions submit -c obscura -f "outputs/$NAME/submission.csv" -m "$MSG"
