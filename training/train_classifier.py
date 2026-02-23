"""
Classifier fine-tuning script for AI text detection.

Fine-tunes roberta-base-openai-detector (or specified model) on the prepared
{text, label} dataset for binary classification: Human (0) vs AI (1).

Improvements:
  - Gradient accumulation for effective larger batch sizes
  - Auto fp16 on CUDA
  - Warmup schedule
  - Better early stopping
  - Learning rate scheduling
  - Class-weighted loss to handle imbalanced data
  - Data sanity checks before training

Usage:
    python -m training.train_classifier
    python -m training.train_classifier --epochs 5 --batch_size 16
    python -m training.train_classifier --model microsoft/deberta-v3-large
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODELS_DIR


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for the trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def validate_dataset(ds, split_name: str):
    """Validate a dataset before training — catch label issues early."""
    print(f"\n--- Validating {split_name} dataset ---")
    print(f"  Samples: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Features: {ds.features}")

    # Check labels
    labels = ds["labels"]
    
    # Convert tensor labels to python ints
    if hasattr(labels, "tolist"):
        labels_list = labels.tolist()
    else:
        # HuggingFace datasets might return lazy accessor
        labels_list = [l.item() if hasattr(l, "item") else l for l in labels]
    
    label_counts = Counter(labels_list)
    
    print(f"  Label distribution: {dict(label_counts)}")

    # Verify labels are 0 and 1
    unique_labels = set(label_counts.keys())
    if unique_labels != {0, 1}:
        print(f"  ⚠️  WARNING: Expected labels {{0, 1}}, got {unique_labels}")
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Invalid labels found: {unique_labels}. Expected only 0 and 1.")

    # Check for empty texts
    if "input_ids" in ds.column_names:
        # After tokenization — check for all-padding sequences
        sample_ids = ds[0]["input_ids"]
        print(f"  Token sequence length: {len(sample_ids)}")

    # Show sample
    print(f"\n  Sample label=0 (Human):")
    for i, row in enumerate(ds):
        if row["labels"] == 0:
            tokens = row.get("input_ids", [])
            print(f"    Token count: {sum(1 for t in tokens if t != 0)}")
            break

    print(f"  Sample label=1 (AI):")
    for i, row in enumerate(ds):
        if row["labels"] == 1:
            tokens = row.get("input_ids", [])
            print(f"    Token count: {sum(1 for t in tokens if t != 0)}")
            break

    return label_counts


def train(
    model_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_length: int | None = None,
):
    """
    Fine-tune DeBERTa on the prepared dataset.

    All arguments default to config.TRAINING_CONFIG values.
    """
    model_name = model_name or config.CLASSIFIER_MODEL
    tc = config.TRAINING_CONFIG
    epochs = epochs or tc["epochs"]
    batch_size = batch_size or tc["batch_size"]
    learning_rate = learning_rate or tc["learning_rate"]
    max_length = max_length or tc["max_length"]
    grad_accum = tc.get("gradient_accumulation_steps", 2)
    warmup_ratio = tc.get("warmup_ratio", 0.1)
    weight_decay = tc.get("weight_decay", 0.01)
    patience = tc.get("early_stopping_patience", 3)
    output_dir = str(MODEL_DIR)

    # Auto-detect mixed precision capability
    use_fp16 = tc.get("fp16", False) and torch.cuda.is_available()
    use_bf16 = tc.get("bf16", False) and torch.cuda.is_available()

    # --- Load data ---
    print(f"Loading data from {DATA_DIR}...")
    train_ds = load_from_disk(str(DATA_DIR / "train"))
    test_ds = load_from_disk(str(DATA_DIR / "test"))
    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    # --- Validate raw labels ---
    print("\n--- Pre-tokenization label check ---")
    raw_labels = train_ds["label"]
    raw_counts = Counter(raw_labels)
    print(f"  Raw train label distribution: {dict(raw_counts)}")
    print(f"  Raw label type: {type(raw_labels[0])}")

    # Ensure labels are plain integers
    if hasattr(raw_labels[0], 'item'):
        print("  Labels are tensor type, will convert to int")
    
    # --- Tokenize ---
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    print("Tokenizing datasets...")
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    # --- Validate datasets after processing ---
    train_label_counts = validate_dataset(train_ds, "train")
    validate_dataset(test_ds, "test")

    # Verify a single sample after format conversion
    sample = train_ds[0]
    print(f"\n--- Post-format check ---")
    print(f"  labels dtype: {sample['labels'].dtype}")
    print(f"  labels value: {sample['labels']}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")

    # --- Compute class weights ---
    n_human = train_label_counts.get(0, 0)
    n_ai = train_label_counts.get(1, 0)
    total = n_human + n_ai
    if n_human > 0 and n_ai > 0:
        # Inverse frequency weighting
        w_human = total / (2.0 * n_human)
        w_ai = total / (2.0 * n_ai)
        class_weights = [w_human, w_ai]
        print(f"\n  Class weights: Human={w_human:.4f}, AI={w_ai:.4f}")
    else:
        class_weights = None
        print("\n  ⚠️  WARNING: Cannot compute class weights (missing class)")

    # --- Model ---
    print(f"\nLoading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1},
    )

    effective_batch = batch_size * grad_accum
    total_steps = (len(train_ds) // effective_batch) * epochs

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=3,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        dataloader_num_workers=0,  # Windows compatibility
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    print(f"\n{'='*60}")
    print(f"Starting training:")
    print(f"  Model:           {model_name}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size} (effective: {effective_batch})")
    print(f"  LR:              {learning_rate}")
    print(f"  Warmup ratio:    {warmup_ratio}")
    print(f"  FP16:            {use_fp16}")
    print(f"  BF16:            {use_bf16}")
    print(f"  Max length:      {max_length}")
    print(f"  Total steps:     ~{total_steps}")
    print(f"  Class weights:   {class_weights}")
    print(f"  Device:          {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Output:          {output_dir}")
    print(f"  Early stopping:  patience={patience} on F1")
    print(f"{'='*60}\n")

    trainer.train()

    # --- Save best model ---
    best_dir = str(MODEL_DIR / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nBest model saved to: {best_dir}")

    # --- Final evaluation ---
    print("\nFinal evaluation on test set:")
    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for AI detection")
    parser.add_argument("--model", type=str, default=None, help="Base model name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=None, help="Max token length")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
