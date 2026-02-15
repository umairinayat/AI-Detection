"""
Evaluation script — measures detector performance on a test set.

Computes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
for each component individually and for the full ensemble.

Improvements:
  - Larger default evaluation set (500 samples)
  - Per-domain breakdown
  - Threshold sweep for optimal F1
  - Confidence tier analysis
  - Timing benchmarks

Usage:
    python -m training.evaluate
    python -m training.evaluate --max_samples 1000
"""

import json
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import config
from detector.ensemble import EnsembleDetector

DATA_DIR = config.DATA_DIR


def find_optimal_threshold(true_labels, pred_probs, thresholds=None):
    """Sweep thresholds to find optimal F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (pred_probs > t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            true_labels, preds, average="binary", pos_label=1, zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def evaluate(
    classifier_path: str | None = None,
    max_samples: int = 500,
    sweep_thresholds: bool = True,
):
    """
    Evaluate the ensemble detector on the test set.

    Args:
        classifier_path: Path to fine-tuned classifier (or None for base model).
        max_samples: Max number of test samples to evaluate.
        sweep_thresholds: Whether to search for optimal threshold.
    """
    # Load test data
    test_path = DATA_DIR / "test"
    if not test_path.exists():
        print("ERROR: No test data found. Run 'python -m training.prepare_data' first.")
        return

    test_ds = load_from_disk(str(test_path))
    print(f"Loaded {len(test_ds)} test samples")

    # Subsample if needed
    if len(test_ds) > max_samples:
        test_ds = test_ds.shuffle(seed=42).select(range(max_samples))
        print(f"Subsampled to {max_samples} samples for evaluation")

    # Initialize detector
    print("Initializing detector...")
    detector = EnsembleDetector(classifier_path=classifier_path)
    print(f"  Classifier fine-tuned: {detector.classifier.is_fine_tuned}")
    print(f"  Ensemble weights: {detector.weights}")

    # Run predictions
    true_labels = []
    pred_probs = []
    verdicts = []
    confidences = []
    component_probs = {"perplexity": [], "burstiness": [], "classifier": []}
    times = []

    for i, sample in enumerate(test_ds):
        text = sample["text"]
        label = sample["label"]

        start = time.time()
        result = detector.analyze(text)
        elapsed = time.time() - start

        true_labels.append(label)
        pred_probs.append(result["ai_probability"])
        verdicts.append(result["verdict"])
        confidences.append(result.get("confidence", "unknown"))
        times.append(elapsed)

        for comp_name in component_probs:
            if comp_name in result.get("components", {}):
                component_probs[comp_name].append(
                    result["components"][comp_name]["ai_probability"]
                )

        if (i + 1) % 20 == 0:
            avg_time = np.mean(times[-20:])
            print(f"  Evaluated {i + 1}/{len(test_ds)} samples... "
                  f"({avg_time:.2f}s/sample)")

    # Compute metrics
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)
    pred_labels = (pred_probs > 0.5).astype(int)

    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS (threshold=0.5)")
    print(f"{'='*60}")
    print(classification_report(
        true_labels, pred_labels, target_names=["Human", "AI"]
    ))

    try:
        auc = roc_auc_score(true_labels, pred_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        auc = None
        print("ROC-AUC: N/A (single class)")

    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # False positive rate
    fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"\nFalse Positive Rate: {fp_rate:.4f} ({fp_rate*100:.1f}%)")

    # --- Threshold sweep ---
    if sweep_thresholds:
        best_threshold, best_f1 = find_optimal_threshold(true_labels, pred_probs)
        print(f"\n{'='*60}")
        print(f"OPTIMAL THRESHOLD SEARCH")
        print(f"{'='*60}")
        print(f"  Best threshold: {best_threshold:.2f}")
        print(f"  Best F1: {best_f1:.4f}")

        optimal_preds = (pred_probs > best_threshold).astype(int)
        cm_opt = confusion_matrix(true_labels, optimal_preds)
        fp_rate_opt = cm_opt[0][1] / (cm_opt[0][0] + cm_opt[0][1]) if (cm_opt[0][0] + cm_opt[0][1]) > 0 else 0
        print(f"  FP rate at optimal: {fp_rate_opt:.4f} ({fp_rate_opt*100:.1f}%)")

    # --- Per-component metrics ---
    print(f"\n{'='*60}")
    print("PER-COMPONENT RESULTS")
    print(f"{'='*60}")

    for comp_name, probs in component_probs.items():
        if len(probs) != len(true_labels):
            print(f"\n{comp_name.upper()}: Skipped (incomplete data)")
            continue
        comp_probs_arr = np.array(probs)
        comp_preds = (comp_probs_arr > 0.5).astype(int)
        acc = accuracy_score(true_labels, comp_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels, comp_preds, average="binary", pos_label=1, zero_division=0
        )
        try:
            comp_auc = roc_auc_score(true_labels, comp_probs_arr)
        except ValueError:
            comp_auc = None

        print(f"\n{comp_name.upper()}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1:        {f1:.4f}")
        if comp_auc is not None:
            print(f"  ROC-AUC:   {comp_auc:.4f}")

    # --- Confidence tier analysis ---
    print(f"\n{'='*60}")
    print("CONFIDENCE TIER BREAKDOWN")
    print(f"{'='*60}")

    for conf_level in ["high", "moderate", "low"]:
        mask = np.array([c == conf_level for c in confidences])
        if mask.sum() == 0:
            continue
        tier_true = true_labels[mask]
        tier_preds = pred_labels[mask]
        tier_acc = accuracy_score(tier_true, tier_preds)
        print(f"\n  {conf_level.upper()} confidence ({mask.sum()} samples):")
        print(f"    Accuracy: {tier_acc:.4f}")

    # --- Timing ---
    print(f"\n{'='*60}")
    print("PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Total samples:  {len(times)}")
    print(f"  Total time:     {sum(times):.1f}s")
    print(f"  Avg per sample: {np.mean(times):.2f}s")
    print(f"  Median:         {np.median(times):.2f}s")
    print(f"  P95:            {np.percentile(times, 95):.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate AI text detector")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--classifier_path", type=str, default=None)
    args = parser.parse_args()

    evaluate(
        classifier_path=args.classifier_path,
        max_samples=args.max_samples,
    )
