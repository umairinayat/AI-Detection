"""
Perplexity Engine — Core detection component.

Uses GPT-2 as a proxy model to measure how "predictable" text is.
AI-generated text has low perplexity (model is unsurprised).
Human text has high perplexity (model encounters unexpected tokens).

Improvements over basic implementation:
  - Sliding window for long documents (no more 512-token truncation)
  - Batched per-sentence computation
  - Configurable sigmoid steepness
  - Per-token surprisal for word-level highlighting
"""

import math
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import config


class PerplexityEngine:
    """Calculates token-level and sentence-level perplexity using GPT-2."""

    def __init__(self, model_name: str | None = None):
        model_name = model_name or config.PERPLEXITY_MODEL
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of a text string using sliding window
        for texts longer than the model's context window.

        PPL = exp( -1/N * Σ log P(x_i | x_<i) )

        For long texts, uses a sliding window approach where only the
        non-overlapping tokens in each window contribute to the loss,
        avoiding double-counting in overlapping regions.

        Returns:
            Perplexity score. Lower = more predictable = more likely AI.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        seq_len = input_ids.size(1)

        if seq_len < 2:
            return float("inf")

        max_length = config.PERPLEXITY_WINDOW
        stride = config.PERPLEXITY_STRIDE

        # Short text: single pass
        if seq_len <= max_length:
            return self._single_pass_ppl(input_ids)

        # Long text: sliding window with proper overlap handling
        total_nll = 0.0
        num_tokens = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            chunk_ids = input_ids[:, begin:end]

            with torch.no_grad():
                outputs = self.model(chunk_ids)
                logits = outputs.logits

            # Compute per-token cross-entropy loss (no reduction)
            # Shift: logits[:-1] predict labels[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Only count non-overlapping tokens to avoid double-counting.
            # First window: count all tokens.
            # Subsequent windows: only count tokens from stride onward
            # (the first `max_length - stride` tokens overlap with the
            # previous window and were already counted).
            if begin == 0:
                # First window: count all token losses
                total_nll += token_losses.sum().item()
                num_tokens += token_losses.size(0)
            else:
                # Subsequent windows: only count the non-overlapping tail.
                # The overlap is (max_length - stride) tokens of context.
                # In the shifted loss array, token at index j corresponds to
                # predicting chunk position j+1 given positions 0..j.
                # We want to count predictions for positions that are new
                # (not covered by the previous window), which start at
                # the overlap boundary.
                overlap = max_length - stride
                # In the loss array (which is shifted by 1), the loss at
                # index i covers predicting token at chunk position i+1.
                # New tokens start at chunk position `overlap`, so we need
                # losses from index `overlap - 1` onward.
                start_idx = max(overlap - 1, 0)
                new_losses = token_losses[start_idx:]
                total_nll += new_losses.sum().item()
                num_tokens += new_losses.size(0)

            if end >= seq_len:
                break

        avg_nll = total_nll / num_tokens if num_tokens > 0 else 0
        return math.exp(avg_nll)

    def _single_pass_ppl(self, input_ids: torch.Tensor) -> float:
        """Single forward pass perplexity for short texts."""
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        return math.exp(loss)

    def calculate_per_sentence_batched(self, sentences: list[str]) -> list[float]:
        """
        Batch-compute perplexity for multiple sentences.

        On GPU with enough sentences, uses padded batches with attention
        masks and computes per-sample loss for true parallel inference.
        Falls back to sequential computation on CPU or small batches.
        """
        if not sentences:
            return []

        # For small batches or CPU, sequential is fine
        if len(sentences) <= 3 or self.device.type == "cpu":
            return [self.calculate_perplexity(s) for s in sentences]

        # True batch processing for GPU
        results = []
        batch_size = 8

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_ppls = self._batch_ppl(batch)
            results.extend(batch_ppls)

        return results

    def _batch_ppl(self, texts: list[str]) -> list[float]:
        """
        Compute perplexity for a batch of texts in parallel on GPU.

        Uses left-padding with attention masks so each sequence's loss
        is computed independently within the batch.
        """
        # Tokenize with padding (left-pad for causal LM)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=config.PERPLEXITY_WINDOW,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batch_size, seq_len = input_ids.shape

        if seq_len < 2:
            return [float("inf")] * batch_size

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        # Per-token cross-entropy (no reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        # Reshape for loss computation: (batch * seq, vocab) vs (batch * seq)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(batch_size, -1)

        # Mask out padding tokens and compute per-sample average NLL
        masked_loss = per_token_loss * shift_mask
        token_counts = shift_mask.sum(dim=-1)  # non-padding token count per sample

        ppls = []
        for j in range(batch_size):
            n_tokens = token_counts[j].item()
            if n_tokens < 1:
                ppls.append(float("inf"))
            else:
                avg_nll = masked_loss[j].sum().item() / n_tokens
                ppls.append(math.exp(avg_nll))

        # Restore default padding side
        self.tokenizer.padding_side = "right"
        return ppls

    def calculate_per_sentence(self, sentences: list[str]) -> list[float]:
        """Calculate perplexity for each sentence. Uses batching when possible."""
        return self.calculate_per_sentence_batched(sentences)

    def get_token_surprisals(self, text: str) -> list[dict]:
        """
        Get per-token surprisal (negative log probability) for word-level highlighting.

        Returns list of {"token": str, "surprisal": float} dicts.
        High surprisal = unexpected = more human-like.
        Low surprisal = expected = more AI-like.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.PERPLEXITY_MAX_TOKENS,
        )
        input_ids = inputs["input_ids"].to(self.device)

        if input_ids.size(1) < 2:
            return []

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute per-token log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Convert to surprisal (negative log prob)
        surprisals = -token_log_probs[0].cpu().numpy()

        # Decode tokens
        token_ids = input_ids[0, 1:].cpu().tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return [
            {"token": tok, "surprisal": float(surp)}
            for tok, surp in zip(tokens, surprisals)
        ]

    def ppl_to_ai_probability(self, ppl: float) -> float:
        """
        Convert perplexity to AI probability using sigmoid mapping.

        Low PPL → high AI probability
        High PPL → low AI probability

        Uses configurable steepness (k) from config.
        """
        if not math.isfinite(ppl):
            return 0.5  # Uncertain for degenerate cases

        threshold = config.PERPLEXITY_THRESHOLD_AI
        k = config.PERPLEXITY_SIGMOID_K
        ai_probability = 1.0 / (1.0 + math.exp(k * (ppl - threshold)))
        return max(0.0, min(1.0, ai_probability))

    def score_text(self, text: str, sentences: list[str]) -> dict:
        """
        Full perplexity analysis.

        Returns dict with:
            - global_ppl: perplexity of the full text
            - sentence_ppls: per-sentence perplexity scores
            - ai_probability: normalized score [0, 1] where 1 = likely AI
            - token_surprisals: per-token surprisal for word-level highlighting
        """
        global_ppl = self.calculate_perplexity(text)
        sentence_ppls = self.calculate_per_sentence(sentences)
        ai_probability = self.ppl_to_ai_probability(global_ppl)

        return {
            "global_ppl": global_ppl,
            "sentence_ppls": sentence_ppls,
            "ai_probability": ai_probability,
        }
