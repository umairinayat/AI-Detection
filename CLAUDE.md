# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt  # additional API deps (fastapi, uvicorn, etc.)

# Streamlit UI
streamlit run app.py

# FastAPI server (use server.py as entrypoint to avoid name clash with api/ package)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# Docs auto-generated at http://localhost:8000/docs

# CLI analysis
python analyze_text.py --text "Your text here"
python analyze_text.py --file document.txt

# Tests
python -m pytest tests/test_detector.py -v          # full unit tests
python -m pytest tests/test_detector.py::TestEnsembleDetector -v  # single class
python tests/smoke_test.py                           # quick end-to-end sanity check

# Training
python training/train_classifier.py    # fine-tune RoBERTa or Qwen2.5-3B (QLoRA)
python training/evaluate.py            # evaluate a checkpoint
```

## API Authentication

API keys are stored in `.api_keys.json` (not committed). The `X-API-Key` header is required for `/analyze`, `/analyze/batch`, and `/analyze/file`. Health and info endpoints are unauthenticated. See `api/auth.py` for key management.

Env vars that affect behavior at startup (see `config.py`):
- `CLASSIFIER_PATH` — override classifier checkpoint path
- `HF_MODEL_REPO` / `HF_MODEL_REVISION` — HuggingFace Hub model (defaults to `umairinayat/llm_detector`)
- `HF_USE_HUB_LLM=1` — force Hub model even when local `best/` checkpoint exists
- `API_HOST`, `API_PORT`

## Architecture

The detector is a **GPTZero-style ensemble** of three components, blended into a single `ai_probability ∈ [0,1]`.

```
Text Input
  └─► preprocessor.py     → full_text, sentences[], paragraphs[], homoglyphs_detected
        ├─► perplexity.py  → global_ppl, sentence_ppls[], ai_probability (sigmoid)
        ├─► burstiness.py  → coefficient_of_variation, ai_probability (sigmoid)
        └─► classifier.py  → document_ai_prob, sentence_ai_probs[], token attributions
              └─► token_attribution.py  (Integrated Gradients per sentence)
  └─► ensemble.py          → weighted blend → verdict, confidence, sentences[]
```

### What "sentence-level" vs "overall" means

| Layer | Field | Produced in |
|-------|-------|-------------|
| **Overall** | `ai_probability` — primary document score | `EnsembleDetector.analyze()` |
| **Overall** | `verdict` — `"Human"`, `"AI"`, `"Mixed"`, `"Unknown"` | Same; Mixed uses sentence stats in gray zone |
| **Per sentence** | `ai_probability` — for UI highlighting | `EnsembleDetector._compute_sentence_scores()` |
| **Per sentence** | `perplexity`, `ppl_ai_probability`, `classifier_prob` | Same list items |

Burstiness is **document-level only** (variance of sentence perplexities — not re-run per sentence). Sentence scores blend per-sentence PPL → AI prob with per-sentence classifier output (60% classifier / 40% PPL when fine-tuned; PPL-only otherwise).

### Core files

1. **`detector/ensemble.py`** — `analyze()` orchestrates everything; `_compute_sentence_scores()` builds the `sentences` list; `_determine_verdict()` uses `config.VERDICT_*` thresholds.
2. **`detector/classifier.py`** — `score_text()` → `document_ai_prob` + `sentence_ai_probs[]` + token attributions. LLM path uses contextual sentence scoring.
3. **`detector/perplexity.py`** — `ppl_to_ai_probability()` sigmoid shared between global and sentence scoring.
4. **`detector/preprocessor.py`** — `preprocess()` sentence list drives all downstream metrics.
5. **`config.py`** — all thresholds and weights; `ENSEMBLE_WEIGHTS_TRAINED/UNTRAINED`, `VERDICT_*`, `CONFIDENCE_THRESHOLDS`.

### Classifier checkpoint resolution order (`config._resolve_classifier_checkpoint`)

1. `CLASSIFIER_PATH` env var
2. `HF_USE_HUB_LLM=1` → `HF_MODEL_REPO`
3. Local `models/detector/llm_detector/best/` (Qwen2.5-3B QLoRA)
4. Local `models/detector/best/` (RoBERTa)
5. `HF_MODEL_REPO` fallback

When no fine-tuned checkpoint is found, `classifier.is_fine_tuned = False` and ensemble uses `ENSEMBLE_WEIGHTS_UNTRAINED` (classifier weight = 0).

### API shape

- `POST /api/v1/analyze` — single text; returns `ai_probability`, `verdict`, `confidence`, `confidence_category`, `components`, `sentences[]`, `metadata`.
- `POST /api/v1/analyze/batch` — up to 10 texts, parallel via `ThreadPoolExecutor`.
- `POST /api/v1/analyze/file` — upload `.txt`, `.pdf`, `.docx` (max 10 MB).
- `GET /api/v1/health` / `GET /api/v1/info` — no auth required.
- Pydantic schemas: `api/models.py`. Route handlers: `api/endpoints.py`.

**`server.py` vs `api.py`**: `api.py` is the FastAPI app; `server.py` is a thin ASGI shim that loads `api.py` by file path to avoid shadowing the `api/` package when using `uvicorn server:app`.

## Conventions

- **PPL mapping stays shared**: `PerplexityEngine.ppl_to_ai_probability` is used for both global and per-sentence scoring — change both together or neither.
- **Sentence blend weights** live in `_compute_sentence_scores` (`clf_weight`/`ppl_weight`); **document blend** in `analyze()` via `config.ENSEMBLE_WEIGHTS_*`.
- **Short/empty text**: `preprocess` may yield no sentences; `analyze` returns `_empty_result()` with `verdict: "Unknown"`.
- New detection features (calibration, new verdict rules, burstiness-inspired per-sentence signals) go in `ensemble.py` + `config.py` to keep behavior centralized.
- Training config for RoBERTa: `config.TRAINING_CONFIG`. For Qwen2.5-3B QLoRA: `config.LLM_TRAINING_CONFIG`. Training datasets defined in `config.DATA_SOURCES`.
