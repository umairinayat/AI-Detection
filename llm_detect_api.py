"""
LLM Detector API
================
FastAPI application that uses umairinayat/llm_detector (PEFT/LoRA on Qwen2.5-3B-Instruct)
to detect AI-generated text at sentence level and overall.

Runs on port 7000.
Serves the web frontend at /
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# suppress verbose transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── NLTK punkt tokenizer ────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

# ── Configuration ────────────────────────────────────────────────────────────
_LOCAL_MODEL   = Path(__file__).resolve().parent / "models" / "llm_detector"
HF_MODEL_ID    = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "umairinayat/llm_detector"
HF_TOKEN       = os.getenv("HF_TOKEN", "")
MAX_TOKEN_LEN  = 512
BATCH_SIZE     = 4  # sentences per inference batch (CPU-friendly)

# ── Global model state ───────────────────────────────────────────────────────
_model      = None
_tokenizer  = None
_device     = None


def load_model():
    """
    Load the PEFT/LoRA classifier using AutoPeftModelForSequenceClassification.

    On GPU  → 4-bit quantization via BitsAndBytes (fast, low VRAM).
    On CPU  → float16 with low_cpu_mem_usage=True; the OS swap file handles
              any memory spill (slow but correct with ≥8 GB swap).
    """
    global _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")
    print(f"Loading model from {HF_MODEL_ID} ...")

    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoTokenizer

    # ── Tokenizer ──────────────────────────────────────────────────────────
    _tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID, token=HF_TOKEN or None, trust_remote_code=True
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    common_kwargs = dict(
        num_labels=2,
        trust_remote_code=True,
        token=HF_TOKEN or None,
        low_cpu_mem_usage=True,
    )

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        _model = AutoPeftModelForSequenceClassification.from_pretrained(
            HF_MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            **common_kwargs,
        )
    else:
        # CPU path — INT8 quantization via `quanto` reduces the model from
        # ~12 GB (float32) down to ~3 GB, fitting in RAM.
        # Inference is slower than GPU but fully functional.
        from transformers import QuantoConfig
        quanto_cfg = QuantoConfig(weights="int8")
        print("  Note: no GPU — using CPU with INT8 quantization (~3 GB).")
        _model = AutoPeftModelForSequenceClassification.from_pretrained(
            HF_MODEL_ID,
            quantization_config=quanto_cfg,
            device_map="cpu",
            **common_kwargs,
        )

    _model.eval()
    # Ensure pad_token_id is set
    if _model.config.pad_token_id is None:
        _model.config.pad_token_id = _tokenizer.pad_token_id

    print("  ✓ Model loaded and ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Detector API",
    description="Sentence-level AI text detection using umairinayat/llm_detector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    text: str
    sentences: dict[str, float]
    scores: dict[str, float]


# ── Inference helpers ─────────────────────────────────────────────────────────
def _score_batch(texts: list[str]) -> list[float]:
    """
    Run a batch of texts through the classifier.
    Returns a list of AI-probability floats (0-1).
    """
    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKEN_LEN,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        # quanto INT8 models compute in bfloat16; autocast aligns dtypes.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = _model(**inputs).logits           # shape (B, 2)
        probs  = F.softmax(logits.to(torch.float32), dim=-1)
        return probs[:, 1].cpu().tolist()              # index 1 = AI class


def _score_all_sentences(sentences: list[str]) -> list[float]:
    """Batch all sentences, respecting BATCH_SIZE."""
    all_probs: list[float] = []
    for i in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[i : i + BATCH_SIZE]
        all_probs.extend(_score_batch(chunk))
    return all_probs


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    Analyse text for AI-generated content.

    Returns per-sentence AI scores (0-100 %) and overall AI/Human scores.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text field cannot be empty")

    # Sentence splitting
    raw_sentences = sent_tokenize(text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    if not sentences:
        raise HTTPException(status_code=400, detail="No scoreable sentences found in text")

    # Inference
    probs = _score_all_sentences(sentences)

    # Build sentence-level dict  { sentence_text: ai_percentage }
    sentence_scores: dict[str, float] = {
        sent: round(prob * 100, 10)
        for sent, prob in zip(sentences, probs)
    }

    # Overall scores
    ai_overall    = sum(sentence_scores.values()) / len(sentence_scores)
    human_overall = 100.0 - ai_overall

    return DetectResponse(
        text=request.text,
        sentences=sentence_scores,
        scores={
            "aioverall":    round(ai_overall,    10),
            "humanoverall": round(human_overall, 10),
        },
    )


@app.get("/health")
async def health():
    return {
        "status":   "healthy" if _model is not None else "loading",
        "model":    HF_MODEL_ID,
        "device":   str(_device),
    }


# ── Static frontend ───────────────────────────────────────────────────────────
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(_static_dir / "index.html"))


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "llm_detect_api:app",
        host="0.0.0.0",
        port=7000,
        reload=False,
        log_level="info",
    )
