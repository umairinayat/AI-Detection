"""
Token Attribution Engine — GPTZero-style word-level highlighting.

Uses Integrated Gradients (IG) to compute per-token attribution scores
that explain why the classifier predicts AI or Human for each sentence.

Positive attribution → token pushes prediction toward AI
Negative attribution → token pushes prediction toward Human

Key features:
  - Integrated Gradients with configurable interpolation steps
  - Sub-word to word aggregation (handles RoBERTa's BPE tokenization)
  - Top-K most influential AI/Human indicator tokens
  - Batch processing for multiple sentences
"""

import torch
import torch.nn.functional as F
import numpy as np

import config


class TokenAttributor:
    """
    Computes token-level attribution scores using Integrated Gradients.

    Given a classifier model and tokenizer, explains which tokens
    contribute most to the AI vs Human prediction.
    """

    def __init__(self, model, tokenizer, device=None):
        """
        Args:
            model: Fine-tuned RoBERTa/DeBERTa model for sequence classification
            tokenizer: Corresponding tokenizer
            device: torch device (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.num_steps = getattr(config, "ATTRIBUTION_STEPS", 50)
        self.top_k = getattr(config, "ATTRIBUTION_TOP_K", 5)
        self.highlight_threshold = getattr(config, "TOKEN_HIGHLIGHT_THRESHOLD", 0.05)

    @torch.no_grad()
    def _get_prediction(self, input_ids, attention_mask):
        """Get model prediction probability for AI class."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)
        # AI class is typically index 1 for roberta-base-openai-detector
        # but for some models it might be index 0, we handle both
        if probs.shape[-1] == 2:
            return probs[0, 1].item()  # P(AI)
        return probs[0, 0].item()

    def _integrated_gradients(self, input_ids, attention_mask, target_class=1):
        """
        Compute Integrated Gradients for the target class.

        IG formula: (x - x') * integral(dF/dx along path from x' to x)

        Where x = actual input embeddings, x' = baseline (zero) embeddings.

        Args:
            input_ids: Token IDs [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            target_class: Class to attribute (1 = AI)

        Returns:
            attributions: numpy array of shape [seq_len] with per-token scores
        """
        self.model.eval()

        # Get the embedding layer
        if hasattr(self.model, "roberta"):
            embeddings = self.model.roberta.embeddings.word_embeddings
        elif hasattr(self.model, "deberta"):
            embeddings = self.model.deberta.embeddings.word_embeddings
        elif hasattr(self.model, "base_model"):
            embeddings = self.model.base_model.embeddings.word_embeddings
        else:
            raise ValueError("Cannot find embedding layer in model architecture")

        # Get actual embeddings
        actual_embeds = embeddings(input_ids).detach()  # [1, seq_len, embed_dim]

        # Baseline is zero embeddings (PAD token equivalent)
        baseline_embeds = torch.zeros_like(actual_embeds)

        # Accumulate gradients along the path
        accumulated_grads = torch.zeros_like(actual_embeds)

        for step in range(self.num_steps):
            # Interpolate between baseline and actual
            alpha = step / self.num_steps
            interpolated = baseline_embeds + alpha * (actual_embeds - baseline_embeds)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            # Forward pass with interpolated embeddings
            outputs = self.model(
                inputs_embeds=interpolated,
                attention_mask=attention_mask,
            )

            # Get target class logit
            logits = outputs.logits
            target_logit = logits[0, target_class]

            # Backward pass
            self.model.zero_grad()
            target_logit.backward()

            # Accumulate gradients
            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad.detach()

        # Average gradients and multiply by (actual - baseline) = actual
        avg_grads = accumulated_grads / self.num_steps
        attributions = (actual_embeds - baseline_embeds) * avg_grads

        # Sum across embedding dimension to get per-token attribution
        token_attributions = attributions.sum(dim=-1).squeeze(0)  # [seq_len]

        return token_attributions.cpu().numpy()

    def _aggregate_subword_attributions(self, tokens, attributions):
        """
        Merge sub-word token attributions into whole-word attributions.

        RoBERTa uses BPE with 'Ġ' prefix for word-starting tokens.
        DeBERTa uses '▁' prefix.

        Args:
            tokens: List of string tokens from tokenizer
            attributions: Array of per-token attribution scores

        Returns:
            List of (word, attribution_score) tuples
        """
        word_attributions = []
        current_word = ""
        current_score = 0.0
        token_count = 0

        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            # Skip special tokens
            if token in ("<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"):
                continue

            # Check if this token starts a new word
            is_word_start = (
                token.startswith("Ġ")  # RoBERTa BPE prefix
                or token.startswith("▁")  # SentencePiece prefix
                or i == 0  # First non-special token
                or (i > 0 and tokens[i - 1] in ("<s>", "[CLS]"))
            )

            if is_word_start and current_word:
                # Save previous word
                avg_score = current_score / max(token_count, 1)
                clean_word = current_word.strip()
                if clean_word:
                    word_attributions.append((clean_word, float(avg_score)))
                current_word = ""
                current_score = 0.0
                token_count = 0

            # Clean token text
            clean_token = token.replace("Ġ", "").replace("▁", "")
            current_word += clean_token
            current_score += attr
            token_count += 1

        # Don't forget the last word
        if current_word.strip():
            avg_score = current_score / max(token_count, 1)
            word_attributions.append((current_word.strip(), float(avg_score)))

        return word_attributions

    def compute_attributions(self, text, target_class=None):
        """
        Compute token-level attribution scores for a piece of text.

        Args:
            text: Input text string
            target_class: Class to explain (None = auto-detect from prediction).
                          1 = explain AI prediction, 0 = explain Human prediction.

        Returns:
            dict with:
                - word_attributions: List of (word, score) tuples
                - top_ai_tokens: Top K tokens pushing toward AI
                - top_human_tokens: Top K tokens pushing toward Human
                - predicted_class: Model's predicted class (0=Human, 1=AI)
                - predicted_prob: Model's predicted probability for AI class
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=config.CLASSIFIER_MAX_TOKENS,
            truncation=True,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get model prediction first
        ai_prob = self._get_prediction(input_ids, attention_mask)
        predicted_class = 1 if ai_prob > 0.5 else 0

        # Determine which class to attribute
        if target_class is None:
            target_class = predicted_class

        # Compute Integrated Gradients
        raw_attributions = self._integrated_gradients(
            input_ids, attention_mask, target_class=target_class
        )

        # Get tokens for mapping
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Aggregate sub-word attributions to word-level
        word_attributions = self._aggregate_subword_attributions(tokens, raw_attributions)

        # If we attributed the Human class (0), flip signs so positive = AI
        if target_class == 0:
            word_attributions = [(w, -s) for w, s in word_attributions]

        # Normalize attributions to [-1, 1] range
        if word_attributions:
            max_abs = max(abs(s) for _, s in word_attributions) or 1.0
            word_attributions = [(w, s / max_abs) for w, s in word_attributions]

        # Get top K AI and Human indicator tokens
        sorted_by_ai = sorted(word_attributions, key=lambda x: x[1], reverse=True)
        sorted_by_human = sorted(word_attributions, key=lambda x: x[1])

        top_ai_tokens = [
            (w, round(s, 4))
            for w, s in sorted_by_ai[:self.top_k]
            if s > self.highlight_threshold
        ]
        top_human_tokens = [
            (w, round(s, 4))
            for w, s in sorted_by_human[:self.top_k]
            if s < -self.highlight_threshold
        ]

        return {
            "word_attributions": [(w, round(s, 4)) for w, s in word_attributions],
            "top_ai_tokens": top_ai_tokens,
            "top_human_tokens": top_human_tokens,
            "predicted_class": predicted_class,
            "predicted_prob": round(ai_prob, 4),
        }

    def compute_batch_attributions(self, sentences):
        """
        Compute attributions for a list of sentences.

        Args:
            sentences: List of text strings

        Returns:
            List of attribution dicts (one per sentence)
        """
        results = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                # Too short for meaningful attribution
                results.append({
                    "word_attributions": [],
                    "top_ai_tokens": [],
                    "top_human_tokens": [],
                    "predicted_class": -1,
                    "predicted_prob": 0.5,
                })
            else:
                try:
                    result = self.compute_attributions(sentence)
                    results.append(result)
                except Exception as e:
                    # Graceful degradation — return empty if attribution fails
                    results.append({
                        "word_attributions": [],
                        "top_ai_tokens": [],
                        "top_human_tokens": [],
                        "predicted_class": -1,
                        "predicted_prob": 0.5,
                        "error": str(e),
                    })
        return results
