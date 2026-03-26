"""
Text Classifier — Fine-tuned transformer for AI text detection.

Supports two backends:
  1. LLM (Qwen2.5-3B + QLoRA) — loaded if models/llm_detector/best/ exists
  2. RoBERTa (roberta-base-openai-detector) — fallback fine-tuned checkpoint

Key features:
  - Properly guards against untrained model (returns 0.5 = uncertain)
  - 4-bit quantization for LLM inference (bitsandbytes)
  - Batched sentence-level inference
  - Contextual sentence classification (passes surrounding context)
  - Token-level attribution via Integrated Gradients (GPTZero-style)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config
from detector.token_attribution import TokenAttributor


def _hub_revision(path: str) -> str | None:
    """Use HF_MODEL_REVISION when loading from Hub (repo id), not for local dirs."""
    from pathlib import Path

    if "/" in path and not Path(path).exists():
        return config.HF_MODEL_REVISION
    return None


def _is_peft_checkpoint(path: str) -> bool:
    """Check if a checkpoint is a PEFT/LoRA adapter checkpoint (local or HF Hub)."""
    from pathlib import Path
    # Check local path first
    if Path(path).is_dir() and (Path(path) / "adapter_config.json").exists():
        return True
    # Check if it looks like a HF Hub repo ID (e.g. "user/repo")
    if "/" in path and not Path(path).exists():
        try:
            from huggingface_hub import hf_hub_download

            kwargs: dict = {"repo_id": path, "filename": "adapter_config.json"}
            rev = _hub_revision(path)
            if rev:
                kwargs["revision"] = rev
            hf_hub_download(**kwargs)
            return True
        except Exception:
            return False
    return False


class TextClassifier:
    """Binary classifier: Human (0) vs AI (1)."""

    def __init__(self, model_path: str | None = None):
        """
        Load the classifier model.

        Priority:
          1. LLM checkpoint (PEFT LoRA) — from models/llm_detector/best/
          2. RoBERTa fine-tuned checkpoint — from models/detector/best/
          3. Base RoBERTa model (returns 0.5 if not a pre-trained detector)
        """
        resolved_path = model_path or config.CLASSIFIER_CHECKPOINT
        self._is_llm = False

        if resolved_path and _is_peft_checkpoint(resolved_path):
            # ── LLM QLoRA checkpoint ─────────────────────────────────────
            _rev = _hub_revision(resolved_path)
            _rev_kw = {"revision": _rev} if _rev else {}
            print(f"Loading fine-tuned LLM classifier (QLoRA) from {resolved_path}...")
            if _rev:
                print(f"  (Hub revision {_rev[:12]}… — read-only, no upload)")
            try:
                from peft import PeftModel, PeftConfig
                from transformers import BitsAndBytesConfig

                peft_cfg = PeftConfig.from_pretrained(resolved_path, **_rev_kw)
                base_model_name = peft_cfg.base_model_name_or_path

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_name,
                    num_labels=2,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.model = PeftModel.from_pretrained(
                    base_model, resolved_path, **_rev_kw
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    resolved_path, trust_remote_code=True, **_rev_kw
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.model.config.pad_token_id is None:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
                # Also ensure the base model config has it, to prevent the ValueError
                if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config") and self.model.base_model.config.pad_token_id is None:
                    self.model.base_model.config.pad_token_id = self.tokenizer.pad_token_id
                self._is_fine_tuned = True
                self._is_llm = True
                print("  ✓ LLM QLoRA classifier loaded")
            except Exception as e:
                print(f"  ⚠️  Failed to load LLM checkpoint: {e}. Falling back to RoBERTa.")
                resolved_path = config._roberta_checkpoint  # type: ignore[attr-defined]
                self._is_llm = False

        if not self._is_llm:
            if resolved_path:
                # ── RoBERTa fine-tuned ────────────────────────────────────
                print(f"Loading fine-tuned classifier from {resolved_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    resolved_path, num_labels=2
                )
                self._is_fine_tuned = True
            else:
                # ── Base model ────────────────────────────────────────────
                print(f"Loading base model: {config.CLASSIFIER_MODEL}...")
                self.tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.CLASSIFIER_MODEL,
                    num_labels=2,
                    id2label={0: "Human", 1: "AI"},
                    label2id={"Human": 0, "AI": 1},
                )
                if "openai-detector" in config.CLASSIFIER_MODEL or "chatgpt-detector" in config.CLASSIFIER_MODEL:
                    print("  ✓ Model is a pre-trained detector (enabling predictions)")
                    self._is_fine_tuned = True
                else:
                    self._is_fine_tuned = False

        if not self._is_llm:
            # LLM uses device_map="auto"; RoBERTa needs manual placement
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-init token attributor (created on first use)
        self._attributor = None

    @property
    def is_fine_tuned(self) -> bool:
        """Whether a fine-tuned checkpoint is loaded (vs. base model)."""
        return self._is_fine_tuned

    def get_model_and_tokenizer(self):
        """Expose model and tokenizer for external use (e.g. TokenAttributor)."""
        return self.model, self.tokenizer

    @property
    def attributor(self) -> TokenAttributor:
        """Lazy-initialized TokenAttributor instance."""
        if self._attributor is None:
            self._attributor = TokenAttributor(
                self.model, self.tokenizer, self.device
            )
        return self._attributor

    def predict_single(self, text: str) -> float:
        """
        Predict AI probability for a single text.

        Returns:
            Float in [0, 1] — probability that the text is AI-generated.
            Returns 0.5 (uncertain) if model is not fine-tuned.
        """
        # Guard: untrained model produces noise, return uncertain
        if not self._is_fine_tuned:
            return 0.5

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.CLASSIFIER_MAX_TOKENS,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            ai_prob = probs[0][1].item()  # Index 1 = AI class

        return ai_prob

    def predict_sentence_with_context(
        self, sentence: str, prev_sentence: str = "", next_sentence: str = ""
    ) -> float:
        """
        Predict AI probability for a sentence WITH surrounding context.
        Passing context improves accuracy by giving the model more signal.

        Returns:
            Float in [0, 1] — probability that the sentence is AI-generated.
        """
        if not self._is_fine_tuned:
            return 0.5

        # Build contextual input: [prev] <SEP> target <SEP> [next]
        parts = []
        if prev_sentence:
            parts.append(prev_sentence)
        parts.append(sentence)
        if next_sentence:
            parts.append(next_sentence)

        contextual_text = " ".join(parts)
        return self.predict_single(contextual_text)

    def predict_batch(self, texts: list[str]) -> list[float]:
        """
        Batch predict AI probabilities for multiple texts.
        More efficient than calling predict_single repeatedly on GPU.
        """
        if not self._is_fine_tuned:
            return [0.5] * len(texts)

        if not texts:
            return []

        batch_size = 16
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=config.CLASSIFIER_MAX_TOKENS,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                ai_probs = probs[:, 1].cpu().tolist()

            all_probs.extend(ai_probs)

        return all_probs

    def _contextual_inputs(self, sentences: list[str]) -> list[str]:
        """
        Build contextual input for each sentence: prev + sentence + next.

        Passing surrounding context helps the model distinguish AI sentences
        that happen to use casual language, and human sentences in formal text.
        This is the same technique GPTZero uses for per-sentence scoring.
        """
        inputs = []
        for i, sent in enumerate(sentences):
            prev = sentences[i - 1] if i > 0 else ""
            nxt  = sentences[i + 1] if i < len(sentences) - 1 else ""
            ctx  = " ".join(filter(None, [prev, sent, nxt]))
            inputs.append(ctx)
        return inputs

    def score_text(self, text: str, sentences: list[str]) -> dict:
        """
        GPTZero-style scoring: score each sentence with context, then
        derive the document score as a length-weighted mean of sentence scores.

        Returns dict with:
            - document_ai_prob: length-weighted mean of sentence scores
            - sentence_ai_probs: per-sentence AI probabilities [0, 1]
            - ai_probability: same as document_ai_prob (used by ensemble)
            - is_fine_tuned: whether a fine-tuned checkpoint is loaded
            - sentence_attributions: per-sentence token attribution data
        """
        _empty_attr = {
            "word_attributions": [],
            "top_ai_tokens": [],
            "top_human_tokens": [],
            "predicted_class": -1,
            "predicted_prob": 0.5,
        }

        if not self._is_fine_tuned or not sentences:
            return {
                "document_ai_prob": 0.5,
                "sentence_ai_probs": [0.5] * len(sentences),
                "ai_probability": 0.5,
                "is_fine_tuned": self._is_fine_tuned,
                "sentence_attributions": [_empty_attr.copy() for _ in sentences],
            }

        # Score each sentence with surrounding context (GPTZero-style)
        contextual = self._contextual_inputs(sentences)
        sentence_ai_probs = self.predict_batch(contextual)

        # Document score = length-weighted mean of sentence scores.
        # Longer sentences carry more signal than short fragments.
        weights = [max(len(s.split()), 1) for s in sentences]
        total_w = sum(weights)
        document_ai_prob = sum(p * w for p, w in zip(sentence_ai_probs, weights)) / total_w

        # Token-level attributions (run on raw sentences, not contextual inputs,
        # so highlighting aligns with the displayed sentence text)
        if self._is_fine_tuned and sentences:
            sentence_attributions = self.attributor.compute_batch_attributions(sentences)
        else:
            sentence_attributions = [_empty_attr.copy() for _ in sentences]

        return {
            "document_ai_prob": document_ai_prob,
            "sentence_ai_probs": sentence_ai_probs,
            "ai_probability": document_ai_prob,
            "is_fine_tuned": self._is_fine_tuned,
            "sentence_attributions": sentence_attributions,
        }
