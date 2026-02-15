"""
Burstiness Analyzer — Second-order detection signal.

Measures the *variance* of sentence-level perplexity.
Human writing has "bursty" perplexity — complex sentences followed by simple ones.
AI writing has flat, consistent perplexity across sentences.

Enhanced with multiple variance features:
  - Coefficient of variation (primary)
  - Interquartile range ratio
  - Lag-1 autocorrelation (sequential pattern)
  - Max/min range ratio
"""

import numpy as np

import config


class BurstinessAnalyzer:
    """Analyzes the temporal variation of perplexity across sentences."""

    def score(self, sentence_ppls: list[float]) -> dict:
        """
        Compute burstiness from per-sentence perplexity scores.

        Args:
            sentence_ppls: List of perplexity values, one per sentence.

        Returns dict with:
            - burstiness: coefficient of variation of sentence PPLs
            - ppl_std: standard deviation of sentence PPLs
            - ppl_mean: mean of sentence PPLs
            - ai_probability: normalized [0, 1] score where 1 = likely AI
            - features: detailed variance features dict
        """
        if len(sentence_ppls) < 2:
            # Can't compute variance with fewer than 2 sentences
            return {
                "burstiness": 0.0,
                "ppl_std": 0.0,
                "ppl_mean": sentence_ppls[0] if sentence_ppls else 0.0,
                "ai_probability": 0.5,  # Uncertain
                "features": {},
            }

        ppls = np.array(sentence_ppls, dtype=np.float64)

        # Filter out infinities (from very short sentences)
        ppls = ppls[np.isfinite(ppls)]
        if len(ppls) < 2:
            return {
                "burstiness": 0.0,
                "ppl_std": 0.0,
                "ppl_mean": 0.0,
                "ai_probability": 0.5,
                "features": {},
            }

        ppl_mean = float(np.mean(ppls))
        ppl_std = float(np.std(ppls))
        ppl_median = float(np.median(ppls))

        # --- Feature 1: Coefficient of variation (primary burstiness metric) ---
        burstiness = ppl_std / ppl_mean if ppl_mean > 1e-6 else 0.0

        # --- Feature 2: Interquartile range ratio ---
        q75, q25 = np.percentile(ppls, [75, 25])
        iqr_ratio = (q75 - q25) / ppl_median if ppl_median > 1e-6 else 0.0

        # --- Feature 3: Range ratio (max/min) ---
        ppl_min = float(np.min(ppls))
        ppl_max = float(np.max(ppls))
        range_ratio = (ppl_max - ppl_min) / ppl_mean if ppl_mean > 1e-6 else 0.0

        # --- Feature 4: Lag-1 autocorrelation (sequential pattern) ---
        # High autocorrelation means PPL changes gradually (more AI-like)
        # Low/negative means PPL jumps around (more human-like)
        if len(ppls) >= 3:
            lag1_corr = float(np.corrcoef(ppls[:-1], ppls[1:])[0, 1])
            if not np.isfinite(lag1_corr):
                lag1_corr = 0.0
        else:
            lag1_corr = 0.0

        # --- Feature 5: Median absolute deviation (robust to outliers) ---
        mad = float(np.median(np.abs(ppls - ppl_median)))
        mad_ratio = mad / ppl_median if ppl_median > 1e-6 else 0.0

        # --- Combined AI probability ---
        # Primary: sigmoid on coefficient of variation
        threshold = config.BURSTINESS_THRESHOLD_AI
        k = config.BURSTINESS_SIGMOID_K
        cov_prob = 1.0 / (1.0 + np.exp(k * (burstiness - threshold)))

        # Secondary: IQR-based probability (low IQR = AI-like)
        iqr_prob = 1.0 / (1.0 + np.exp(3.0 * (iqr_ratio - 0.5)))

        # Combined (CoV is primary, IQR is secondary confirmation)
        ai_probability = float(0.7 * cov_prob + 0.3 * iqr_prob)
        ai_probability = max(0.0, min(1.0, ai_probability))

        features = {
            "coefficient_of_variation": burstiness,
            "iqr_ratio": float(iqr_ratio),
            "range_ratio": float(range_ratio),
            "lag1_autocorrelation": lag1_corr,
            "mad_ratio": float(mad_ratio),
            "ppl_median": ppl_median,
            "ppl_min": ppl_min,
            "ppl_max": ppl_max,
        }

        return {
            "burstiness": burstiness,
            "ppl_std": ppl_std,
            "ppl_mean": ppl_mean,
            "ai_probability": ai_probability,
            "features": features,
        }
