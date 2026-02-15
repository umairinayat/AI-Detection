"""
Unit tests for the AI text detection engine.

Tests each component individually with known AI and human text samples.
Updated for new features: confidence tiers, homoglyph normalization,
dynamic ensemble weights, sliding window perplexity.

Usage:
    python -m pytest tests/test_detector.py -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Test fixtures ---

# Known AI-generated text (typical ChatGPT style — smooth, predictable)
AI_TEXT = (
    "Artificial intelligence has made significant strides in recent years, "
    "transforming various industries and reshaping the way we interact with "
    "technology. Machine learning algorithms have become increasingly sophisticated, "
    "enabling computers to process vast amounts of data and derive meaningful insights. "
    "These advancements have led to breakthroughs in natural language processing, "
    "computer vision, and autonomous systems. As we continue to push the boundaries "
    "of what is possible, it is essential to consider the ethical implications of "
    "these technologies and ensure they are developed responsibly."
)

# Known human-written text (varied, creative, irregular pacing)
HUMAN_TEXT = (
    "I've been thinking about this for weeks now, and honestly? The whole AI thing "
    "scares the crap out of me. Not because robots are going to take over — that's "
    "movie stuff. But because... well, have you tried talking to ChatGPT? It sounds "
    "way too confident about everything. My professor last semester couldn't even tell "
    "the difference. She gave an A to a paper that was basically just a prompt. "
    "Wild times we're living in. Absolutely wild."
)

SHORT_TEXT = "Hello world."


class TestPreprocessor:
    """Test text preprocessing."""

    def test_normalize_whitespace(self):
        from detector.preprocessor import normalize_text
        result = normalize_text("  hello   world  \n\n  test  ")
        assert result == "hello world test"

    def test_remove_zero_width_chars(self):
        from detector.preprocessor import normalize_text
        result = normalize_text("hello\u200bworld\u200c")
        assert result == "helloworld"

    def test_remove_soft_hyphens(self):
        from detector.preprocessor import normalize_text
        result = normalize_text("soft\u00adhyphen")
        assert result == "softhyphen"

    def test_homoglyph_normalization(self):
        from detector.preprocessor import normalize_unicode
        # Cyrillic 'A' (U+0410) should be normalized to Latin 'A'
        result = normalize_unicode("\u0410BC")
        assert result == "ABC"

    def test_homoglyph_detection(self):
        from detector.preprocessor import preprocess
        # Text with Cyrillic 'A' (U+0410) in mostly-English context
        result = preprocess("\u0410rtificial intelligence is a field of study.")
        assert result["homoglyphs_detected"] is True

    def test_homoglyph_no_false_positive_for_non_english(self):
        from detector.preprocessor import preprocess
        # Cyrillic text (Russian) — should NOT flag homoglyphs
        result = preprocess(
            "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! "
            "\u042d\u0442\u043e \u0442\u0435\u043a\u0441\u0442 \u043d\u0430 "
            "\u0440\u0443\u0441\u0441\u043a\u043e\u043c \u044f\u0437\u044b\u043a\u0435."
        )
        assert result["homoglyphs_detected"] is False

    def test_split_sentences(self):
        from detector.preprocessor import split_sentences
        sentences = split_sentences(AI_TEXT)
        assert len(sentences) >= 3

    def test_split_filters_short(self):
        from detector.preprocessor import split_sentences
        sentences = split_sentences("Hi. Ok. This is a longer sentence that should pass.")
        # Only sentences above the minimum length should pass
        assert all(len(s) > 10 for s in sentences)

    def test_preprocess_returns_full_dict(self):
        from detector.preprocessor import preprocess
        result = preprocess(AI_TEXT)
        assert "full_text" in result
        assert "sentences" in result
        assert "paragraphs" in result
        assert "language" in result
        assert "original_length" in result
        assert "homoglyphs_detected" in result
        assert isinstance(result["sentences"], list)

    def test_language_detection(self):
        from detector.preprocessor import detect_language
        assert detect_language("This is an English sentence.") == "en"

    def test_paragraph_splitting(self):
        from detector.preprocessor import split_paragraphs
        text = "First paragraph here, it is long enough.\n\nSecond paragraph here, also long enough."
        paragraphs = split_paragraphs(text)
        assert len(paragraphs) == 2


class TestPerplexityEngine:
    """Test the perplexity calculation engine."""

    @pytest.fixture(scope="class")
    def engine(self):
        from detector.perplexity import PerplexityEngine
        return PerplexityEngine()

    def test_returns_positive_float(self, engine):
        ppl = engine.calculate_perplexity(AI_TEXT)
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_ai_text_lower_ppl_than_human(self, engine):
        """Core detection hypothesis: AI text should have lower perplexity."""
        ai_ppl = engine.calculate_perplexity(AI_TEXT)
        human_ppl = engine.calculate_perplexity(HUMAN_TEXT)
        assert ai_ppl < human_ppl, (
            f"Expected AI PPL ({ai_ppl:.2f}) < Human PPL ({human_ppl:.2f})"
        )

    def test_score_text_structure(self, engine):
        from detector.preprocessor import preprocess
        processed = preprocess(AI_TEXT)
        result = engine.score_text(processed["full_text"], processed["sentences"])
        assert "global_ppl" in result
        assert "sentence_ppls" in result
        assert "ai_probability" in result
        assert 0.0 <= result["ai_probability"] <= 1.0

    def test_short_text_handled(self, engine):
        ppl = engine.calculate_perplexity(SHORT_TEXT)
        assert isinstance(ppl, float)

    def test_ppl_to_ai_probability(self, engine):
        """Low PPL should give high AI probability."""
        low_ppl_prob = engine.ppl_to_ai_probability(10.0)
        high_ppl_prob = engine.ppl_to_ai_probability(80.0)
        assert low_ppl_prob > high_ppl_prob

    def test_infinite_ppl_returns_uncertain(self, engine):
        prob = engine.ppl_to_ai_probability(float("inf"))
        assert prob == 0.5


class TestBurstinessAnalyzer:
    """Test the burstiness analyzer."""

    def test_low_variance_input(self):
        from detector.burstiness import BurstinessAnalyzer
        analyzer = BurstinessAnalyzer()
        result = analyzer.score([10.0, 11.0, 10.5, 10.2, 10.8])
        assert result["burstiness"] < 0.5

    def test_high_variance_input(self):
        from detector.burstiness import BurstinessAnalyzer
        analyzer = BurstinessAnalyzer()
        result = analyzer.score([5.0, 50.0, 8.0, 100.0, 12.0, 80.0])
        assert result["burstiness"] > 0.5

    def test_single_sentence(self):
        from detector.burstiness import BurstinessAnalyzer
        analyzer = BurstinessAnalyzer()
        result = analyzer.score([15.0])
        assert result["ai_probability"] == 0.5

    def test_score_structure(self):
        from detector.burstiness import BurstinessAnalyzer
        analyzer = BurstinessAnalyzer()
        result = analyzer.score([10.0, 20.0, 15.0])
        assert "burstiness" in result
        assert "ppl_std" in result
        assert "ppl_mean" in result
        assert "ai_probability" in result
        assert "features" in result

    def test_features_present(self):
        from detector.burstiness import BurstinessAnalyzer
        analyzer = BurstinessAnalyzer()
        result = analyzer.score([10.0, 20.0, 15.0, 30.0, 5.0])
        features = result["features"]
        assert "coefficient_of_variation" in features
        assert "iqr_ratio" in features
        assert "lag1_autocorrelation" in features
        assert "mad_ratio" in features


class TestClassifier:
    """Test the DeBERTa classifier."""

    @pytest.fixture(scope="class")
    def classifier(self):
        from detector.classifier import TextClassifier
        return TextClassifier()

    def test_untrained_returns_uncertain(self, classifier):
        """Base model should return 0.5 for all inputs (uncertain)."""
        if not classifier.is_fine_tuned:
            prob = classifier.predict_single(AI_TEXT)
            assert prob == 0.5

    def test_batch_prediction(self, classifier):
        probs = classifier.predict_batch([AI_TEXT, HUMAN_TEXT])
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_score_text_structure(self, classifier):
        from detector.preprocessor import preprocess
        processed = preprocess(AI_TEXT)
        result = classifier.score_text(processed["full_text"], processed["sentences"])
        assert "ai_probability" in result
        assert "sentence_ai_probs" in result
        assert "is_fine_tuned" in result


class TestEnsembleDetector:
    """Test the full ensemble pipeline."""

    @pytest.fixture(scope="class")
    def detector(self):
        from detector.ensemble import EnsembleDetector
        return EnsembleDetector()

    def test_analyze_returns_full_structure(self, detector):
        result = detector.analyze(AI_TEXT)
        assert "verdict" in result
        assert "ai_probability" in result
        assert "confidence" in result
        assert "confidence_category" in result
        assert "components" in result
        assert "sentences" in result
        assert "metadata" in result
        assert result["verdict"] in ("Human", "AI", "Mixed", "Unknown")

    def test_probability_in_range(self, detector):
        result = detector.analyze(AI_TEXT)
        assert 0.0 <= result["ai_probability"] <= 1.0

    def test_empty_text(self, detector):
        result = detector.analyze("")
        assert result["verdict"] == "Unknown"

    def test_sentences_have_scores(self, detector):
        result = detector.analyze(AI_TEXT)
        for s in result["sentences"]:
            assert "text" in s
            assert "perplexity" in s
            assert "ai_probability" in s
            assert "ppl_ai_probability" in s
            assert "classifier_prob" in s

    def test_confidence_tiers(self, detector):
        result = detector.analyze(AI_TEXT)
        assert result["confidence"] in ("high", "moderate", "low")
        assert len(result["confidence_category"]) > 0

    def test_metadata_present(self, detector):
        result = detector.analyze(AI_TEXT)
        metadata = result["metadata"]
        assert "num_sentences" in metadata
        assert "language" in metadata
        assert "classifier_trained" in metadata
        assert "ensemble_weights" in metadata

    def test_dynamic_weights_when_untrained(self, detector):
        """When classifier is not fine-tuned, its weight should be 0."""
        if not detector.classifier.is_fine_tuned:
            assert detector.weights["classifier"] == 0.0
