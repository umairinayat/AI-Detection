"""
Centralized configuration for the AI Text Detection system.
All thresholds, model paths, and ensemble weights are defined here.

Architecture mirrors GPTZero's multi-component approach:
  - Perplexity analysis (proxy LM)
  - Burstiness analysis (sentence-level variance)
  - Deep learning classifier (fine-tuned transformer)
  - Token-level attribution (Integrated Gradients)
  - Weighted ensemble with confidence scoring
"""

import os
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "detector"
LLM_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "llm_checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# --- HuggingFace Hub Configuration ---
HF_MODEL_REVISION: str | None = os.getenv("HF_MODEL_REVISION", None)
HF_USE_HUB_LLM: bool = os.getenv("HF_USE_HUB_LLM", "0").strip() in ("1", "true", "yes")

# --- Model Configuration ---
PERPLEXITY_MODEL = "gpt2"  # CPU-friendly model for inference (fast)
# DeBERTa-v3-large: best model for sentence-level AI detection
# Outperforms Qwen2.5 QLoRA on this task: discriminative (not generative),
# calibrated probabilities, runs on 4 GB VRAM, no quantization needed.
CLASSIFIER_MODEL = "microsoft/deberta-v3-large"
CLASSIFIER_MAX_TOKENS = 256  # Sentences are short; 256 is plenty and 2x faster

# --- Token Attribution Configuration ---
# Integrated Gradients for GPTZero-style word-level highlighting
ATTRIBUTION_METHOD = "integrated_gradients"
ATTRIBUTION_STEPS = 50  # Number of interpolation steps for IG (higher = more precise, slower)
TOKEN_HIGHLIGHT_THRESHOLD = 0.05  # Min |attribution| score to highlight a word
ATTRIBUTION_TOP_K = 5  # Number of top AI/Human indicator tokens to return per sentence

# --- Classifier Checkpoint Resolution ---
# Priority: CLASSIFIER_PATH env var → local fine-tuned best/ → None (loads base model)
# The old Qwen/HF hub model is intentionally NOT in this chain.
_best_checkpoint = MODELS_DIR / "best"
_classifier_path_override = os.getenv("CLASSIFIER_PATH", "").strip()


def _resolve_classifier_checkpoint() -> str | None:
    """Load from explicit env var, then local fine-tuned checkpoint, then None (base model)."""
    if _classifier_path_override:
        return _classifier_path_override
    if _best_checkpoint.exists():
        return str(_best_checkpoint)
    return None


CLASSIFIER_CHECKPOINT = _resolve_classifier_checkpoint()

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
# Fine-tuned DeBERTa at sentence level is the sole signal (GPTZero-style).
# Perplexity and burstiness are kept at 0 — the classifier already captures
# those signals and adds far more discriminative power.
ENSEMBLE_WEIGHTS_TRAINED = {
    "perplexity": 0.0,
    "burstiness": 0.0,
    "classifier": 1.0,
}

# When classifier is NOT fine-tuned: fall back to perplexity + burstiness
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

# --- Training Configuration (DeBERTa-v3-large, 500k sentences, 12 GB VRAM) ---
#
# gradient_checkpointing=True is what makes large fit on 12 GB.
# It recomputes activations during backward pass instead of storing them —
# cuts VRAM by ~40% at the cost of ~20% slower training. Worth it.
#
# Effective batch = 4 * 16 = 64 (same as larger GPUs, just more accumulation steps).
TRAINING_CONFIG = {
    "epochs": 1,
    "batch_size": 2,                   # reduced from 4 to prevent OOM on 12 GB
    "gradient_accumulation_steps": 32, # effective batch = 64
    "learning_rate": 5e-6,          # optimal for DeBERTa-v3-large fine-tuning (lowered to prevent NaN)
    # Cap training rows after load_from_disk (shuffle+select). None = use full train split.
    # Set to e.g. 500_000 if you built a larger on-disk set but want a fixed random subset.
    "max_train_samples": 300000,
    "warmup_ratio": 0.05,           # 5% warmup over 315k * 3 epochs
    "weight_decay": 0.01,
    "max_length": 256,              # sentences fit in 256; saves 2x memory vs 512
    "label_smoothing": 0.05,        # KEY: prevents flatline at 0.99/0.01
    "gradient_checkpointing": False, # KEY: Disable gradient checkpointing to prevent NaN in Deberta-v3
    "fp16": False,
    "bf16": True,                   # Enabled to properly scale DeBERTa's native float16 weights and prevent NaN overflow
    "output_dir": str(MODELS_DIR),
    "early_stopping_patience": 2,
    "metric_for_best_model": "f1",
    "eval_steps": 1000,             # evaluate every 1000 optimizer steps
    "save_steps": 1000,
    "logging_steps": 100,
}

# --- LLM Training Configuration (Qwen2.5-3B + QLoRA) ---
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_TRAINING_CONFIG = {
    "epochs": 1,
    "batch_size": 4,                     # Increased from 2; QLoRA fits 4 on 20GB
    "gradient_accumulation_steps": 8,    # Effective batch = 32
    "learning_rate": 2e-4,               # Higher LR for LoRA adapters
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_length": 256,                   # Optimized for sentence-level
    "bf16": True,
    "output_dir": str(LLM_CHECKPOINT_DIR),
    "early_stopping_patience": 2,
    "metric_for_best_model": "f1",
    "max_train_samples": 300000,          # Subsample to 300k as requested
    "max_eval_samples": 1000,             # Subsample eval as well
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # QLoRA (4-bit) config
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
}

# --- Data Sources ---
# All datasets used for training (expandable)
DATA_SOURCES = {
    "hc3": {
        "name": "Hello-SimpleAI/HC3",
        "description": "Human vs ChatGPT comparison corpus",
        "max_samples": 100000,  # Maximized
    },
    "dmitva": {
        "name": "dmitva/human_ai_generated_text",
        "description": "Large-scale human/AI text pairs",
        "max_samples": 100000,  # Maximized
    },
    "raid": {
        "name": "liamdugan/raid",
        "description": "RAID benchmark: 11 LLMs + adversarial attacks",
        "max_samples": 200000,  # Maximized
    },
    "ai_text_pile": {
        "name": "artem9k/ai-text-detection-pile",
        "description": "Large AI text detection pile",
        "max_samples": 200000,  # Maximized
    },
    "gptwiki": {
        "name": "aadityaubhat/GPT-wiki-intro",
        "description": "Wikipedia intros vs GPT-3.5 rephrasings",
        "max_samples": 50000,
    },
    # --- New datasets ---
    "shankar": {
        "name": "shankarramani/AI_Human_text",
        "description": "General human vs AI writing (diverse domains)",
        "max_samples": 100000,
    },
    "nicola": {
        "name": "NicolaiSivesind/ChatGPT-generated-dataset",
        "description": "Reddit-style human vs ChatGPT",
        "max_samples": 50000,
    },
    "semeval": {
        "name": "symanto/autext23",
        "description": "SemEval 2024-style human vs AI (Wikipedia, Reddit, WikiHow, PeerRead)",
        "max_samples": 50000,
    },
    "m4": {
        "name": "sarvivarma/human_ai_text",
        "description": "M4 corpus: human vs multi-LLM (GPT4, Claude, Llama, etc.)",
        "max_samples": 100000,
    },
    "essays": {
        "name": "qwedsacf/ivypanda-essays",
        "description": "High-quality human essays baseline",
        "max_samples": 30000,
        "human_only": True,  # Only human samples, AI side from RAID/HC3
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
