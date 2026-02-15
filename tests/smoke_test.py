"""Quick smoke test — verifies the full pipeline end-to-end."""

import sys
sys.path.insert(0, ".")

from detector.preprocessor import preprocess
from detector.perplexity import PerplexityEngine

# Test preprocessor
print("=== Preprocessor ===")
result = preprocess(
    "AI has made significant strides in recent years. "
    "Machine learning is transforming industries. "
    "The advancements are remarkable. "
    "I honestly think it's kinda scary though."
)
print(f"Sentences found: {len(result['sentences'])}")
for i, s in enumerate(result["sentences"]):
    print(f"  [{i}] {s}")

# Test perplexity engine
print("\n=== Perplexity Engine ===")
engine = PerplexityEngine()

ai_text = (
    "Artificial intelligence has made significant strides in recent years, "
    "transforming various industries and reshaping the way we interact with technology."
)
human_text = (
    "I've been thinking about this for weeks now, and honestly? The whole AI thing "
    "scares me. My professor couldn't even tell the difference."
)

ai_ppl = engine.calculate_perplexity(ai_text)
human_ppl = engine.calculate_perplexity(human_text)
print(f"AI text PPL:    {ai_ppl:.2f}")
print(f"Human text PPL: {human_ppl:.2f}")
print(f"AI < Human?     {ai_ppl < human_ppl} (expected: True)")

# Test full ensemble
print("\n=== Full Ensemble ===")
from detector.ensemble import EnsembleDetector
detector = EnsembleDetector()

ai_result = detector.analyze(ai_text)
print(f"\nAI text verdict: {ai_result['verdict']} ({ai_result['ai_probability']:.0%})")

human_result = detector.analyze(human_text)
print(f"Human text verdict: {human_result['verdict']} ({human_result['ai_probability']:.0%})")

# Sentence-level
print(f"\nSentence-level analysis (AI text):")
for s in ai_result["sentences"]:
    label = "AI" if s["ai_probability"] > 0.6 else "Human" if s["ai_probability"] < 0.4 else "Uncertain"
    print(f"  [{label}] PPL={s['perplexity']:.1f} | {s['text'][:60]}...")

print("\nSentence-level analysis (Human text):")
for s in human_result["sentences"]:
    label = "AI" if s["ai_probability"] > 0.6 else "Human" if s["ai_probability"] < 0.4 else "Uncertain"
    print(f"  [{label}] PPL={s['perplexity']:.1f} | {s['text'][:60]}...")

print("\n=== SMOKE TEST PASSED ===")
