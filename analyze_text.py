"""
Enhanced AI Text Analysis with Detailed Explanations

Provides sentence-by-sentence detection with explanations of WHY
each sentence is classified as AI or human.

Features:
- Sentence-level detection with probabilities
- Overall AI/Human percentage
- Detailed explanations for each classification
- Word-level indicators (which words suggest AI vs Human)
- Perplexity and burstiness analysis

Usage:
    python analyze_text.py
    python analyze_text.py --file document.txt
    python analyze_text.py --text "Your text here"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.ensemble import EnsembleDetector
from detector.perplexity import PerplexityEngine
import config


class DetailedAnalyzer:
    """Enhanced analyzer with explanations."""

    def __init__(self):
        """Initialize the detector with best available model (see config / .env)."""
        if config.CLASSIFIER_CHECKPOINT:
            print(f"✓ Classifier checkpoint: {config.CLASSIFIER_CHECKPOINT}")
        else:
            print("⚠ No fine-tuned checkpoint; using base RoBERTa")
        self.detector = EnsembleDetector()

        self.perplexity_engine = PerplexityEngine()

    def explain_sentence(self, sentence: str, analysis: Dict) -> Dict[str, Any]:
        """
        Generate detailed explanation for why a sentence is classified as AI or human.

        Args:
            sentence: The sentence text
            analysis: Analysis results from detector

        Returns:
            Dictionary with detailed explanations
        """
        ai_prob = analysis.get("ai_probability", 0.5)
        ppl = analysis.get("perplexity", 0)

        # Determine classification
        is_ai = ai_prob > 0.6
        confidence = "High" if (ai_prob > 0.8 or ai_prob < 0.2) else "Moderate" if (ai_prob > 0.65 or ai_prob < 0.35) else "Low"

        # Build explanation
        explanation = {
            "classification": "AI-Generated" if is_ai else "Human-Written",
            "confidence": confidence,
            "ai_probability": f"{ai_prob:.1%}",
            "human_probability": f"{1-ai_prob:.1%}",
            "reasons": []
        }

        # Perplexity-based reasons
        if ppl < 15:
            explanation["reasons"].append({
                "signal": "Perplexity",
                "indicator": "Very Low",
                "value": f"{ppl:.1f}",
                "meaning": "Text is highly predictable - typical of AI models that generate expected word sequences",
                "suggests": "AI"
            })
        elif ppl < 25:
            explanation["reasons"].append({
                "signal": "Perplexity",
                "indicator": "Low",
                "value": f"{ppl:.1f}",
                "meaning": "Text is quite predictable - may be AI-generated or formulaic writing",
                "suggests": "Likely AI"
            })
        elif ppl > 40:
            explanation["reasons"].append({
                "signal": "Perplexity",
                "indicator": "High",
                "value": f"{ppl:.1f}",
                "meaning": "Text contains unexpected word choices - characteristic of human creativity",
                "suggests": "Human"
            })
        else:
            explanation["reasons"].append({
                "signal": "Perplexity",
                "indicator": "Moderate",
                "value": f"{ppl:.1f}",
                "meaning": "Text predictability is in the uncertain range",
                "suggests": "Uncertain"
            })

        # Sentence characteristics
        words = sentence.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        if word_count > 30:
            explanation["reasons"].append({
                "signal": "Sentence Length",
                "indicator": "Long",
                "value": f"{word_count} words",
                "meaning": "AI often generates longer, more complex sentences",
                "suggests": "AI"
            })
        elif word_count < 8:
            explanation["reasons"].append({
                "signal": "Sentence Length",
                "indicator": "Short",
                "value": f"{word_count} words",
                "meaning": "Very short sentences suggest informal human writing",
                "suggests": "Human"
            })

        # Word complexity
        if avg_word_length > 6:
            explanation["reasons"].append({
                "signal": "Word Complexity",
                "indicator": "High",
                "value": f"{avg_word_length:.1f} chars/word",
                "meaning": "Longer words suggest formal/AI writing",
                "suggests": "AI"
            })

        # Check for AI indicators
        ai_phrases = ["furthermore", "moreover", "however", "additionally", "consequently",
                      "therefore", "thus", "hence", "accordingly", "as a result"]
        human_phrases = ["i think", "i feel", "maybe", "kinda", "gonna", "wanna",
                        "yeah", "nah", "lol", "tbh", "imo"]

        sentence_lower = sentence.lower()
        found_ai_phrases = [p for p in ai_phrases if p in sentence_lower]
        found_human_phrases = [p for p in human_phrases if p in sentence_lower]

        if found_ai_phrases:
            explanation["reasons"].append({
                "signal": "Formal Connectors",
                "indicator": "Present",
                "value": ", ".join(found_ai_phrases),
                "meaning": "Formal transition words common in AI-generated text",
                "suggests": "AI"
            })

        if found_human_phrases:
            explanation["reasons"].append({
                "signal": "Informal Language",
                "indicator": "Present",
                "value": ", ".join(found_human_phrases),
                "meaning": "Casual/colloquial expressions typical of human writing",
                "suggests": "Human"
            })

        return explanation

    def analyze_text_detailed(self, text: str) -> Dict[str, Any]:
        """
        Perform detailed analysis with explanations.

        Args:
            text: Text to analyze

        Returns:
            Comprehensive analysis results with explanations
        """
        print("\n" + "="*70)
        print("🔍 AI TEXT DETECTION - DETAILED ANALYSIS")
        print("="*70)

        # Run detector
        result = self.detector.analyze(text)

        # Calculate overall statistics
        sentences = result.get("sentences", [])
        total_sentences = len(sentences)

        ai_sentences = sum(1 for s in sentences if s["ai_probability"] > 0.6)
        human_sentences = sum(1 for s in sentences if s["ai_probability"] < 0.4)
        uncertain_sentences = total_sentences - ai_sentences - human_sentences

        ai_percentage = (ai_sentences / total_sentences * 100) if total_sentences > 0 else 0
        human_percentage = (human_sentences / total_sentences * 100) if total_sentences > 0 else 0
        uncertain_percentage = (uncertain_sentences / total_sentences * 100) if total_sentences > 0 else 0

        # Print overall verdict
        print(f"\n📊 OVERALL VERDICT: {result['verdict']}")
        print(f"   AI Probability: {result['ai_probability']:.1%}")
        print(f"   Confidence: {result['confidence'].upper()}")
        print(f"   {result['confidence_category']}")

        # Print statistics
        print(f"\n📈 SENTENCE-LEVEL STATISTICS:")
        print(f"   Total Sentences: {total_sentences}")
        print(f"   🔴 AI Sentences: {ai_sentences} ({ai_percentage:.1f}%)")
        print(f"   🟢 Human Sentences: {human_sentences} ({human_percentage:.1f}%)")
        print(f"   🟡 Uncertain: {uncertain_sentences} ({uncertain_percentage:.1f}%)")

        # Component breakdown
        print(f"\n🔬 COMPONENT ANALYSIS:")
        comp = result["components"]
        print(f"   Perplexity Score: {comp['perplexity']['global_ppl']:.1f}")
        print(f"   → AI Probability: {comp['perplexity']['ai_probability']:.1%}")
        print(f"   Burstiness Score: {comp['burstiness']['score']:.3f}")
        print(f"   → AI Probability: {comp['burstiness']['ai_probability']:.1%}")
        if comp['classifier']['is_fine_tuned']:
            print(f"   Classifier Score: {comp['classifier']['ai_probability']:.1%}")
        else:
            print(f"   Classifier: Not trained (using base models only)")

        # Sentence-by-sentence analysis
        print(f"\n📝 SENTENCE-BY-SENTENCE ANALYSIS:\n")
        print("="*70)

        for i, sent_data in enumerate(sentences, 1):
            sent_text = sent_data["text"]

            # Get detailed explanation
            explanation = self.explain_sentence(sent_text, sent_data)

            # Print sentence header
            icon = "🔴" if sent_data["ai_probability"] > 0.6 else "🟢" if sent_data["ai_probability"] < 0.4 else "🟡"
            print(f"\n{icon} SENTENCE {i}: {explanation['classification']} ({explanation['confidence']} Confidence)")
            print(f"   AI: {explanation['ai_probability']} | Human: {explanation['human_probability']}")
            print(f"\n   Text: \"{sent_text}\"")

            # Print reasons
            if explanation["reasons"]:
                print(f"\n   💡 Why this classification:")
                for reason in explanation["reasons"]:
                    print(f"      • {reason['signal']}: {reason['indicator']} ({reason['value']})")
                    print(f"        → {reason['meaning']}")
                    print(f"        → Suggests: {reason['suggests']}")

            # Print token attributions (GPTZero-style word indicators)
            top_ai = sent_data.get("top_ai_tokens", [])
            top_human = sent_data.get("top_human_tokens", [])
            if top_ai or top_human:
                print(f"\n   🔍 Token Attribution (GPTZero-style):")
                if top_ai:
                    tokens_str = ", ".join(f'"{w}" ({s:+.2f})' for w, s in top_ai)
                    print(f"      🔴 AI indicators: {tokens_str}")
                if top_human:
                    tokens_str = ", ".join(f'"{w}" ({s:+.2f})' for w, s in top_human)
                    print(f"      🟢 Human indicators: {tokens_str}")

            print("-"*70)

        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")

        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Detailed AI text detection with explanations")
    parser.add_argument("--text", type=str, help="Text to analyze (quoted string)")
    parser.add_argument("--file", type=str, help="Path to text file to analyze")

    args = parser.parse_args()

    # Get text from arguments or prompt user
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    else:
        print("\n" + "="*70)
        print("Enter or paste your text (press Ctrl+Z then Enter on Windows, or Ctrl+D on Mac/Linux when done):")
        print("="*70 + "\n")
        text = sys.stdin.read()

    if not text or not text.strip():
        print("❌ Error: No text provided")
        sys.exit(1)

    # Run analysis
    analyzer = DetailedAnalyzer()
    analyzer.analyze_text_detailed(text)


if __name__ == "__main__":
    main()
