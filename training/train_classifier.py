"""
Classifier fine-tuning script for AI text detection.

Fine-tunes DeBERTa (config.CLASSIFIER_MODEL, default microsoft/deberta-v3-large)
on the prepared {text, label} dataset for binary classification: Human (0) vs AI (1).
Sentence-level data is expected (see scripts/prepare_sentence_data.py).

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
import logging
import sys
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config

DATA_DIR = config.DATA_DIR
MODEL_DIR = config.MODELS_DIR

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "train_under_all.log"

def _setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Also capture transformers library logs into the same file
    logging.getLogger("transformers").setLevel(logging.INFO)

_setup_logging()
log = logging.getLogger(__name__)
log.info(
    "\n%s\n[train_classifier] RUN START %s\n%s",
    "=" * 60,
    datetime.now().isoformat(timespec="seconds"),
    "=" * 60,
)
log.info(f"Log file (append): {LOG_FILE}")


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
    """Trainer with class-weighted cross-entropy loss and label smoothing.

    label_smoothing (e.g. 0.05) converts hard 0/1 labels to 0.05/0.95.
    This is the primary fix for the flatline-at-1.0 problem: it forces
    the model to produce calibrated probabilities instead of saturating
    the softmax at the extremes.
    """

    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(dtype=logits.dtype, device=logits.device) if self.class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def validate_dataset(ds, split_name: str):
    """Validate a dataset before training — catch label issues early."""
    log.info(f"Validating {split_name}: {len(ds):,} samples  cols={ds.column_names}")

    labels = ds["labels"]
    if hasattr(labels, "tolist"):
        labels_list = labels.tolist()
    else:
        labels_list = [l.item() if hasattr(l, "item") else l for l in labels]

    label_counts = Counter(labels_list)
    log.info(f"  Label distribution: {dict(label_counts)}")

    unique_labels = set(label_counts.keys())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid labels: {unique_labels}. Expected only 0 and 1.")
    if unique_labels != {0, 1}:
        log.warning(f"  Only one class present: {unique_labels}")

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
    use_grad_ckpt = tc.get("gradient_checkpointing", False)

    # --- Load data ---
    log.info(f"Loading data from {DATA_DIR} ...")
    train_ds = load_from_disk(str(DATA_DIR / "train"))
    test_ds  = load_from_disk(str(DATA_DIR / "test"))
    log.info(f"  Train: {len(train_ds):,} samples | Test: {len(test_ds):,} samples")

    max_train = tc.get("max_train_samples")
    if max_train and len(train_ds) > max_train:
        train_ds = train_ds.shuffle(seed=42).select(range(max_train))
        log.info(f"  ↳ Subsampled train to {max_train:,} rows (random, seed=42)")

    # --- Validate raw labels ---
    log.info("Pre-tokenization label check ...")
    raw_labels = train_ds["label"]
    raw_counts = Counter(raw_labels)
    log.info(f"  Raw train label distribution: {dict(raw_counts)}")

    # --- Tokenize ---
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        # No padding here — DataCollatorWithPadding pads dynamically per batch.
        # This is critical for sentence-level data: sentences vary from 8 to 80 words,
        # so static padding to 256 wastes 50-70% of compute on pad tokens.
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    log.info("Tokenizing datasets ...")
    train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=4,
                             remove_columns=["text"])
    test_ds  = test_ds.map(tokenize_fn, batched=True, num_proc=4,
                             remove_columns=["text"])
    train_ds = train_ds.rename_column("label", "labels")
    test_ds  = test_ds.rename_column("label", "labels")

    # --- Validate datasets after processing ---
    train_label_counts = validate_dataset(train_ds, "train")
    validate_dataset(test_ds, "test")

    # --- Compute class weights ---
    n_human = train_label_counts.get(0, 0)
    n_ai    = train_label_counts.get(1, 0)
    total   = n_human + n_ai
    if n_human > 0 and n_ai > 0:
        w_human = total / (2.0 * n_human)
        w_ai    = total / (2.0 * n_ai)
        class_weights = [w_human, w_ai]
        log.info(f"Class weights: Human={w_human:.4f}  AI={w_ai:.4f}")
    else:
        class_weights = None
        log.warning("Cannot compute class weights — missing class in training data")

    # --- Model ---
    log.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1},
    )

    effective_batch = batch_size * grad_accum
    total_steps = (len(train_ds) // effective_batch) * epochs

    # Sub-epoch evaluation: evaluate every eval_steps rather than once per epoch.
    # On 315k train samples this gives ~5 checkpoints per epoch instead of 1.
    eval_steps    = tc.get("eval_steps",    2000)
    save_steps    = tc.get("save_steps",    2000)
    logging_steps = tc.get("logging_steps", 200)

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
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        save_total_limit=3,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_grad_ckpt,
        report_to="none",
        dataloader_num_workers=4,
    )

    label_smoothing = tc.get("label_smoothing", 0.0)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    log.info("=" * 60)
    log.info("Starting training:")
    log.info(f"  Model:              {model_name}")
    log.info(f"  Epochs:             {epochs}")
    log.info(f"  Batch size:         {batch_size}  (effective: {effective_batch})")
    log.info(f"  Learning rate:      {learning_rate}")
    log.info(f"  Warmup ratio:       {warmup_ratio}")
    log.info(f"  Max length:         {max_length}")
    log.info(f"  Grad checkpointing: {use_grad_ckpt}")
    log.info(f"  BF16:               {use_bf16}")
    log.info(f"  Total opt steps:    ~{total_steps:,}")
    log.info(f"  Label smoothing:    {tc.get('label_smoothing', 0.0)}")
    log.info(f"  Save last N ckpts:  3")
    log.info(f"  Device:             {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log.info(f"  Output dir:         {output_dir}")
    log.info(f"  Log file:           {LOG_FILE}")
    log.info("=" * 60)

    trainer.train()

    # --- Save best model ---
    best_dir = str(MODEL_DIR / "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    log.info(f"Best model saved to: {best_dir}")

    # --- Final evaluation ---
    log.info("Final evaluation on test set:")
    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            log.info(f"  {key}: {value:.4f}")

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
