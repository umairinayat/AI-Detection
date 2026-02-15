"""
Centralized configuration for the AI Text Detection system.
All thresholds, model paths, and ensemble weights are defined here.

Architecture mirrors GPTZero's multi-component approach:
  - Perplexity analysis (proxy LM)
  - Burstiness analysis (sentence-level variance)
  - Deep learning classifier (fine-tuned transformer)
  - Weighted ensemble with confidence scoring
"""

import os
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "detector"
DATA_DIR = PROJECT_ROOT / "data"

# --- Model Configuration ---
PERPLEXITY_MODEL = "gpt2"  # CPU-friendly model for inference (fast)
CLASSIFIER_MODEL = "microsoft/deberta-v3-base"  # Base model for fine-tuning
CLASSIFIER_MAX_TOKENS = 512  # Max token length for classifier

# For training on GPU, you can temporarily use larger models:
# CLASSIFIER_MODEL = "microsoft/deberta-v3-large"  # Uncomment for GPU training

# Auto-detect fine-tuned checkpoint
_best_checkpoint = MODELS_DIR / "best"
CLASSIFIER_CHECKPOINT = str(_best_checkpoint) if _best_checkpoint.exists() else None

# --- Perplexity Configuration ---
# AI text typically has PPL ~7-15, human text ~25+
PERPLEXITY_THRESHOLD_AI = 30.0  # Below this → likely AI
PERPLEXITY_MAX_TOKENS = 512  # Max token length for perplexity calculation
PERPLEXITY_SIGMOID_K = 0.15  # Sigmoid steepness (was 0.1, too flat)
PERPLEXITY_STRIDE = 256  # Sliding window stride for long texts
PERPLEXITY_WINDOW = 512  # Sliding window size

# --- Burstiness Configuration ---
# AI text has low burstiness (flat PPL curve), human has high (jagged)
BURSTINESS_THRESHOLD_AI = 0.5  # Coefficient of variation; below this → likely AI
BURSTINESS_SIGMOID_K = 5.0  # Sigmoid steepness

# --- Classifier Thresholds ---
CLASSIFIER_THRESHOLD = 0.5  # Above this → classified as AI

# --- Ensemble Weights ---
# When classifier IS fine-tuned: classifier gets majority weight
ENSEMBLE_WEIGHTS_TRAINED = {
    "perplexity": 0.25,
    "burstiness": 0.20,
    "classifier": 0.55,
}

# When classifier is NOT fine-tuned: exclude it entirely (it produces noise)
ENSEMBLE_WEIGHTS_UNTRAINED = {
    "perplexity": 0.55,
    "burstiness": 0.45,
    "classifier": 0.00,
}

# --- Confidence Tiers (like GPTZero) ---
# Thresholds for confidence categories
CONFIDENCE_THRESHOLDS = {
    "high_ai": 0.85,        # Above this → "Highly confident AI"
    "moderate_ai": 0.65,    # Above this → "Moderately confident AI"
    "uncertain_high": 0.55, # Above this → "Uncertain, leaning AI"
    "uncertain_low": 0.45,  # Above this → "Uncertain"
    "moderate_human": 0.25, # Above this → "Moderately confident Human"
    # Below moderate_human → "Highly confident Human"
}

# --- Verdict Thresholds ---
VERDICT_AI_THRESHOLD = 0.75       # Above → "AI"
VERDICT_HUMAN_THRESHOLD = 0.30    # Below → "Human"
VERDICT_MIXED_AI_SENTENCE = 0.6   # Sentence above this → counted as AI
VERDICT_MIXED_HUMAN_SENTENCE = 0.4  # Sentence below this → counted as Human

# --- Training Configuration ---
TRAINING_CONFIG = {
    "epochs": 8,  # Increased for GPU (was 5)
    "batch_size": 32,  # Increased for 16GB GPU (was 16)
    "gradient_accumulation_steps": 2,  # Effective batch size = 64
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 512,
    "fp16": True,  # FP16 enabled for GPU speed boost
    "output_dir": str(MODELS_DIR),
    "early_stopping_patience": 3,
    "metric_for_best_model": "f1",
}

# --- Data Sources ---
# All datasets used for training (expandable)
DATA_SOURCES = {
    "hc3": {
        "name": "Hello-SimpleAI/HC3",
        "description": "Human vs ChatGPT comparison corpus",
        "max_samples": 10000,
    },
    "dmitva": {
        "name": "dmitva/human_ai_generated_text",
        "description": "Large-scale human/AI text pairs",
        "max_samples": 10000,
    },
    "raid": {
        "name": "liamdugan/raid",
        "description": "RAID benchmark: 11 LLMs + adversarial attacks",
        "max_samples": 100000,  # Increased for GPU training (was 50000)
    },
    "ai_text_pile": {
        "name": "artem9k/ai-text-detection-pile",
        "description": "Large AI text detection pile",
        "max_samples": 100000,  # Increased for GPU training (was 50000)
    },
}

# --- API Configuration ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = 1  # Single worker since model is loaded in memory
API_KEYS_FILE = PROJECT_ROOT / ".api_keys.json"
API_MAX_FILE_SIZE_MB = 10
API_MAX_BATCH_SIZE = 10
API_RATE_LIMIT_PER_MINUTE = 60
