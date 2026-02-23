"""
Ensemble Combiner — Aggregates signals from all detection components.

GPTZero-inspired multi-component detection pipeline:
  - Perplexity (how predictable is the text?)
  - Burstiness (how variable is the perplexity across sentences?)
  - Classifier (what does a fine-tuned transformer say?)
  - Token attribution (which words drive the AI/Human prediction?)

Key improvements:
  - Dynamic weights based on classifier training status
  - Consistent per-sentence probability mapping
  - Confidence tiers (high/moderate/low) like GPTZero
  - Trinary classification (Human/AI/Mixed)
  - Calibrated output mapping (biased toward false negatives)
"""

import math
import numpy as np

import config
from detector.perplexity import PerplexityEngine
from detector.burstiness import BurstinessAnalyzer
from detector.classifier import TextClassifier
from detector.preprocessor import preprocess


class EnsembleDetector:
    """
    Multi-component AI text detector.

    Combines three independent signals:
        1. Perplexity (how predictable is the text?)
        2. Burstiness (how variable is the perplexity across sentences?)
        3. Classifier (what does a fine-tuned DeBERTa say?)
    """

    def __init__(self, classifier_path: str | None = None):
        """
        Initialize all detection components.

        Args:
            classifier_path: Optional path to fine-tuned classifier checkpoint.
        """
        self.perplexity_engine = PerplexityEngine()
        self.burstiness_analyzer = BurstinessAnalyzer()
        self.classifier = TextClassifier(model_path=classifier_path)

        # Use appropriate weights based on classifier status
        if self.classifier.is_fine_tuned:
            self.weights = config.ENSEMBLE_WEIGHTS_TRAINED
        else:
            self.weights = config.ENSEMBLE_WEIGHTS_UNTRAINED

    def analyze(self, text: str) -> dict:
        """
        Run full detection pipeline on input text.

        Returns dict with:
            - verdict: "Human", "AI", or "Mixed"
            - ai_probability: final ensemble score [0, 1]
            - confidence: "high", "moderate", or "low"
            - confidence_category: detailed confidence label
            - components: per-component scores
            - sentences: per-sentence analysis for highlighting
            - metadata: preprocessing metadata
        """
        # Step 1: Preprocess
        processed = preprocess(text)
        full_text = processed["full_text"]
        sentences = processed["sentences"]

        if not sentences:
            return self._empty_result()

        # Step 2: Run each component
        ppl_result = self.perplexity_engine.score_text(full_text, sentences)
        burst_result = self.burstiness_analyzer.score(ppl_result["sentence_ppls"])
        clf_result = self.classifier.score_text(full_text, sentences)

        # Step 3: Weighted ensemble
        w = self.weights
        ensemble_prob = (
            w["perplexity"] * ppl_result["ai_probability"]
            + w["burstiness"] * burst_result["ai_probability"]
            + w["classifier"] * clf_result["ai_probability"]
        )
        ensemble_prob = max(0.0, min(1.0, ensemble_prob))

        # Step 4: Per-sentence scores (consistent sigmoid mapping)
        sentence_scores = self._compute_sentence_scores(
            sentences,
            ppl_result["sentence_ppls"],
            clf_result["sentence_ai_probs"],
            clf_result.get("sentence_attributions", []),
        )

        # Step 5: Verdict + Confidence
        verdict = self._determine_verdict(ensemble_prob, sentence_scores)
        confidence, confidence_category = self._determine_confidence(
            ensemble_prob, verdict
        )

        return {
            "verdict": verdict,
            "ai_probability": round(ensemble_prob, 4),
            "confidence": confidence,
            "confidence_category": confidence_category,
            "components": {
                "perplexity": {
                    "global_ppl": round(ppl_result["global_ppl"], 2),
                    "ai_probability": round(ppl_result["ai_probability"], 4),
                },
                "burstiness": {
                    "score": round(burst_result["burstiness"], 4),
                    "ppl_std": round(burst_result["ppl_std"], 2),
                    "ppl_mean": round(burst_result["ppl_mean"], 2),
                    "ai_probability": round(burst_result["ai_probability"], 4),
                    "features": burst_result.get("features", {}),
                },
                "classifier": {
                    "ai_probability": round(clf_result["ai_probability"], 4),
                    "is_fine_tuned": clf_result["is_fine_tuned"],
                },
            },
            "sentences": sentence_scores,
            "metadata": {
                "original_length": processed.get("original_length", len(text)),
                "num_sentences": len(sentences),
                "language": processed.get("language", "unknown"),
                "homoglyphs_detected": processed.get("homoglyphs_detected", False),
                "classifier_trained": self.classifier.is_fine_tuned,
                "ensemble_weights": dict(self.weights),
            },
        }

    def _compute_sentence_scores(
        self,
        sentences: list[str],
        sentence_ppls: list[float],
        sentence_clf_probs: list[float],
        sentence_attributions: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build per-sentence analysis for UI highlighting.

        Uses the SAME sigmoid mapping as the global perplexity score
        for consistency. Now also includes token-level attribution data.
        """
        results = []
        clf_weight = 0.6 if self.classifier.is_fine_tuned else 0.0
        ppl_weight = 1.0 - clf_weight

        if sentence_attributions is None:
            sentence_attributions = []

        for i, sentence in enumerate(sentences):
            ppl = sentence_ppls[i] if i < len(sentence_ppls) else float("inf")
            clf_prob = sentence_clf_probs[i] if i < len(sentence_clf_probs) else 0.5

            # Use the SAME sigmoid as global perplexity (consistent mapping)
            ppl_prob = self.perplexity_engine.ppl_to_ai_probability(ppl)

            # Blend based on classifier availability
            sentence_ai_prob = ppl_weight * ppl_prob + clf_weight * clf_prob

            sentence_result = {
                "text": sentence,
                "perplexity": round(ppl, 2),
                "ppl_ai_probability": round(ppl_prob, 4),
                "classifier_prob": round(clf_prob, 4),
                "ai_probability": round(sentence_ai_prob, 4),
            }

            # Attach token attribution data if available
            if i < len(sentence_attributions):
                attr = sentence_attributions[i]
                sentence_result["word_attributions"] = attr.get("word_attributions", [])
                sentence_result["top_ai_tokens"] = attr.get("top_ai_tokens", [])
                sentence_result["top_human_tokens"] = attr.get("top_human_tokens", [])
            else:
                sentence_result["word_attributions"] = []
                sentence_result["top_ai_tokens"] = []
                sentence_result["top_human_tokens"] = []

            results.append(sentence_result)
        return results

    def _determine_verdict(
        self, ensemble_prob: float, sentence_scores: list[dict]
    ) -> str:
        """
        Determine overall verdict based on ensemble and sentence-level analysis.

        Trinary classification: Human, AI, or Mixed.
        """
        if ensemble_prob > config.VERDICT_AI_THRESHOLD:
            return "AI"
        elif ensemble_prob < config.VERDICT_HUMAN_THRESHOLD:
            return "Human"

        # Check for mixed authorship: some sentences AI, some human
        ai_sentences = sum(
            1 for s in sentence_scores
            if s["ai_probability"] > config.VERDICT_MIXED_AI_SENTENCE
        )
        human_sentences = sum(
            1 for s in sentence_scores
            if s["ai_probability"] < config.VERDICT_MIXED_HUMAN_SENTENCE
        )
        total = len(sentence_scores)

        # Mixed: at least 20% AI sentences AND 20% human sentences
        if total >= 3 and ai_sentences >= total * 0.2 and human_sentences >= total * 0.2:
            return "Mixed"

        return "AI" if ensemble_prob > 0.5 else "Human"

    def _determine_confidence(
        self, ensemble_prob: float, verdict: str
    ) -> tuple[str, str]:
        """
        Determine confidence level for the prediction.

        Returns:
            (confidence_level, confidence_category)
            confidence_level: "high", "moderate", or "low"
            confidence_category: detailed human-readable label
        """
        thresholds = config.CONFIDENCE_THRESHOLDS

        if ensemble_prob >= thresholds["high_ai"]:
            return "high", "Highly confident this is AI-generated"
        elif ensemble_prob >= thresholds["moderate_ai"]:
            return "moderate", "Moderately confident this is AI-generated"
        elif ensemble_prob >= thresholds["uncertain_high"]:
            return "low", "Uncertain, leaning toward AI-generated"
        elif ensemble_prob >= thresholds["uncertain_low"]:
            return "low", "Uncertain"
        elif ensemble_prob >= thresholds["moderate_human"]:
            return "moderate", "Moderately confident this is human-written"
        else:
            return "high", "Highly confident this is human-written"

    @staticmethod
    def _empty_result() -> dict:
        """Return a default result when no analyzable text is found."""
        return {
            "verdict": "Unknown",
            "ai_probability": 0.0,
            "confidence": "low",
            "confidence_category": "Insufficient text for analysis",
            "components": {},
            "sentences": [],
            "metadata": {},
        }
