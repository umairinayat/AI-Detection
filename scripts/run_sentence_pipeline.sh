#!/usr/bin/env bash
# Build sentence-level data (~5 lakh total = 250k per class by default), then fine-tune
# DeBERTa (see config.CLASSIFIER_MODEL) for one epoch and save to models/detector/best/.
#
# Usage:
#   ./scripts/run_sentence_pipeline.sh
#   TARGET_PER_CLASS=300000 ./scripts/run_sentence_pipeline.sh
#
# Tmux (detached; attach with: tmux attach -t ai-detector-train):
#   tmux new-session -d -s ai-detector-train 'cd /path/to/AI-Detection && ./scripts/run_sentence_pipeline.sh'

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# Prefer explicit PYTHON, then project venv, then active venv, then system python3.
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
else
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: No Python found. Create .venv (python3 -m venv .venv && .venv/bin/pip install -r requirements.txt) or set PYTHON." >&2
  exit 1
fi

# 5 lakhs total ≈ 500,000 sentences: 250k Human + 250k AI (balanced).
TARGET_PER_CLASS="${TARGET_PER_CLASS:-250000}"
RUN_LOG="${RUN_LOG:-$ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S).log}"

{
  echo "=== Pipeline start $(date -Is) ==="
  echo "Repo: $ROOT"
  echo "PYTHON=$PYTHON"
  echo "TARGET_PER_CLASS=$TARGET_PER_CLASS (balanced total ≈ $((TARGET_PER_CLASS * 2)) sentences)"
  echo "Transcript (tee): $RUN_LOG"
  echo ""
  "$PYTHON" scripts/prepare_sentence_data.py --target "$TARGET_PER_CLASS"
  "$PYTHON" -m training.train_classifier
  echo ""
  echo "=== Pipeline end $(date -Is) ==="
} 2>&1 | tee -a "$RUN_LOG"

echo "Transcript saved: $RUN_LOG"
echo "Python logs also append to: $ROOT/logs/train_under_all.log"
