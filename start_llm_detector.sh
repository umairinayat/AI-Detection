#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the LLM Detector API on port 7001
# Model: umairinayat/qwen2.5-3b-ai-text-detector
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

# ── Source HF token from .env ──────────────────────────────────────────────────
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source venv/bin/activate

# ── Kill any existing instance on port 7001 ───────────────────────────────────
pkill -f "uvicorn.*7001" 2>/dev/null || true
sleep 1

echo "──────────────────────────────────────────"
echo "  Starting LLM Detector API on port 7001  "
echo "  Frontend:  http://0.0.0.0:7001/         "
echo "  Docs:      http://0.0.0.0:7001/docs     "
echo "  Health:    http://0.0.0.0:7001/health   "
echo "  Public:    https://tymf2q87c8x8zm-7001.proxy.runpod.net/"
echo "  GPU:       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "──────────────────────────────────────────"

uvicorn llm_detect_api:app \
  --host 0.0.0.0 \
  --port 7001 \
  --log-level info
