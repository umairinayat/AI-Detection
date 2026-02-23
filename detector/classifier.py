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
            hf_hub_download(repo_id=path, filename="adapter_config.json")
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
            print(f"Loading fine-tuned LLM classifier (QLoRA) from {resolved_path}...")
            try:
                from peft import PeftModel, PeftConfig
                from transformers import BitsAndBytesConfig

                peft_cfg = PeftConfig.from_pretrained(resolved_path)
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
                self.model = PeftModel.from_pretrained(base_model, resolved_path)
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
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

    def score_text(self, text: str, sentences: list[str]) -> dict:
        """
        Full classification analysis with token-level attribution.

        Returns dict with:
            - document_ai_prob: AI probability for the full text
            - sentence_ai_probs: per-sentence AI probabilities
            - ai_probability: the document-level score (used by ensemble)
            - is_fine_tuned: whether the model is actually trained
            - sentence_attributions: per-sentence token attribution data
        """
        document_ai_prob = self.predict_single(text)

        # Per-sentence: use batch prediction for efficiency
        if self._is_fine_tuned:
            sentence_ai_probs = self.predict_batch(sentences)
        else:
            sentence_ai_probs = [0.5] * len(sentences)

        # Compute token-level attributions per sentence
        sentence_attributions = []
        if self._is_fine_tuned and sentences:
            sentence_attributions = self.attributor.compute_batch_attributions(
                sentences
            )
        else:
            # Return empty attributions for each sentence
            sentence_attributions = [
                {
                    "word_attributions": [],
                    "top_ai_tokens": [],
                    "top_human_tokens": [],
                    "predicted_class": -1,
                    "predicted_prob": 0.5,
                }
                for _ in sentences
            ]

        return {
            "document_ai_prob": document_ai_prob,
            "sentence_ai_probs": sentence_ai_probs,
            "ai_probability": document_ai_prob,
            "is_fine_tuned": self._is_fine_tuned,
            "sentence_attributions": sentence_attributions,
        }
