#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the LLM Detector API on port 7000
# Model: umairinayat/llm_detector (Qwen2.5-3B + LoRA, INT8 quantized on CPU)
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Ensure swap is active (needed to load the 3B model on low-RAM machines) ──
if ! swapon --show | grep -q swapfile 2>/dev/null; then
  if [ -f /swapfile ]; then
    echo "Activating swap..."
    swapon /swapfile
  else
    echo "Warning: /swapfile not found. Create it with:"
    echo "  sudo dd if=/dev/zero of=/swapfile bs=1G count=8 && sudo mkswap /swapfile && sudo swapon /swapfile"
  fi
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source venv/bin/activate

# ── Source HF token from .env ──────────────────────────────────────────────────
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# ── Kill any existing instance on port 7000 ───────────────────────────────────
pkill -f "uvicorn.*7000" 2>/dev/null || true
sleep 1

echo "──────────────────────────────────────────"
echo "  Starting LLM Detector API on port 7000  "
echo "  Frontend:  http://0.0.0.0:7000/         "
echo "  Docs:      http://0.0.0.0:7000/docs     "
echo "  Health:    http://0.0.0.0:7000/health   "
echo "  NOTE: first request takes ~30 s (CPU)   "
echo "──────────────────────────────────────────"

uvicorn llm_detect_api:app \
  --host 0.0.0.0 \
  --port 7000 \
  --log-level info
