"""
DeBERTa Classifier — Fine-tuned transformer for AI text detection.

Uses microsoft/deberta-v3-base fine-tuned on human/AI labeled pairs.
Provides sentence-level and document-level classification.

Key improvements:
  - Properly guards against untrained model (returns 0.5 = uncertain)
  - Uses CLASSIFIER_MAX_TOKENS instead of PERPLEXITY_MAX_TOKENS
  - Batched sentence-level inference
  - Contextual sentence classification (passes surrounding context)
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config


class TextClassifier:
    """Binary classifier: Human (0) vs AI (1)."""

    def __init__(self, model_path: str | None = None):
        """
        Load the classifier model.

        Args:
            model_path: Path to fine-tuned checkpoint. If None, uses
                        config.CLASSIFIER_CHECKPOINT or falls back to
                        the base model (which will return 0.5 for all inputs).
        """
        resolved_path = model_path or config.CLASSIFIER_CHECKPOINT
        self._is_fine_tuned = resolved_path is not None

        # Use fine-tuned checkpoint if available, otherwise base model
        load_path = resolved_path or config.CLASSIFIER_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path, num_labels=2
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @property
    def is_fine_tuned(self) -> bool:
        """Whether a fine-tuned checkpoint is loaded (vs. base model)."""
        return self._is_fine_tuned

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
        Full classification analysis.

        Returns dict with:
            - document_ai_prob: AI probability for the full text
            - sentence_ai_probs: per-sentence AI probabilities
            - ai_probability: the document-level score (used by ensemble)
            - is_fine_tuned: whether the model is actually trained
        """
        document_ai_prob = self.predict_single(text)

        # Per-sentence: use batch prediction for efficiency
        if self._is_fine_tuned:
            sentence_ai_probs = self.predict_batch(sentences)
        else:
            sentence_ai_probs = [0.5] * len(sentences)

        return {
            "document_ai_prob": document_ai_prob,
            "sentence_ai_probs": sentence_ai_probs,
            "ai_probability": document_ai_prob,
            "is_fine_tuned": self._is_fine_tuned,
        }
