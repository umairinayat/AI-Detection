"""
LLM Classifier Training — Qwen2.5-3B + QLoRA

Fine-tunes Qwen2.5-3B-Instruct as a binary sequence classifier
(Human=0 vs AI=1) using 4-bit QLoRA for memory efficiency.

Strategy:
  - Base model loaded in 4-bit NF4 (bitsandbytes)
  - LoRA adapters injected on attention projections
  - Classification head on top of the pooled last-token representation
  - Saves LoRA adapters + merged model to models/llm_detector/best/
"""

import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = config.DATA_DIR
OUTPUT_DIR = Path(config.LLM_TRAINING_CONFIG["output_dir"])
BEST_DIR   = OUTPUT_DIR / "best"


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
        "f1":        f1_score(labels, preds, average="binary", zero_division=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Weighted Trainer (handles class imbalance)
# ──────────────────────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            )
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def train():
    cfg = config.LLM_TRAINING_CONFIG
    model_name = config.LLM_MODEL

    # ── Load Dataset ────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_DIR}...")
    train_ds = load_from_disk(str(DATA_DIR / "train"))
    test_ds  = load_from_disk(str(DATA_DIR / "test"))
    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    # Subsample if configured (for tractable LLM fine-tuning time)
    max_train = cfg.get("max_train_samples")
    max_eval  = cfg.get("max_eval_samples")
    if max_train and len(train_ds) > max_train:
        train_ds = train_ds.shuffle(seed=42).select(range(max_train))
        print(f"  ↳ Subsampled train to {max_train} samples")
    if max_eval and len(test_ds) > max_eval:
        test_ds = test_ds.shuffle(seed=42).select(range(max_eval))
        print(f"  ↳ Subsampled eval  to {max_eval} samples")

    # Pre-tokenization label check
    raw_labels = train_ds["label"]
    raw_dist   = Counter(raw_labels)
    print(f"\n--- Pre-tokenization label check ---")
    print(f"  Raw train label distribution: {dict(raw_dist)}")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Qwen2.5 uses eos as pad by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-pad for classification

    max_length = cfg["max_length"]

    def tokenize_fn(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,   # DataCollator handles padding
        )
        encoded["labels"] = batch["label"]
        return encoded

    # ── Tokenize ────────────────────────────────────────────────────────────
    print("Tokenizing datasets...")
    cols_to_remove = [c for c in train_ds.column_names if c != "label"]
    train_tok = train_ds.map(
        tokenize_fn, batched=True, remove_columns=cols_to_remove,
        desc="Tokenizing train"
    )
    test_tok = test_ds.map(
        tokenize_fn, batched=True, remove_columns=cols_to_remove,
        desc="Tokenizing test"
    )

    # ── Class weights ───────────────────────────────────────────────────────
    label_list = [int(l) for l in train_ds["label"]]
    n_human = label_list.count(0)
    n_ai    = label_list.count(1)
    n_total = n_human + n_ai

    if n_human > 0 and n_ai > 0:
        w_human = n_total / (2.0 * n_human)
        w_ai    = n_total / (2.0 * n_ai)
        class_weights = [w_human, w_ai]
        print(f"\n  Class weights: {class_weights}")
    else:
        class_weights = None
        print("\n  ⚠️  WARNING: Cannot compute class weights")

    # ── QLoRA Model ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name} (4-bit QLoRA)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # nested quantization for extra savings
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Human", 1: "AI"},
        label2id={"Human": 0, "AI": 1},
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Tie pad token embedding (Qwen2.5 quirk)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for k-bit training (gradient checkpointing, cast norms)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA Config ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training Arguments ──────────────────────────────────────────────────
    total_steps = math.ceil(
        len(train_tok) / (cfg["batch_size"] * cfg["gradient_accumulation_steps"])
    ) * cfg["epochs"]

    print(f"\n{'='*60}")
    print("Starting LLM training:")
    print(f"  Model:           {model_name}")
    print(f"  Epochs:          {cfg['epochs']}")
    print(f"  Batch size:      {cfg['batch_size']} (effective: {cfg['batch_size'] * cfg['gradient_accumulation_steps']})")
    print(f"  LR:              {cfg['learning_rate']}")
    print(f"  BF16:            {cfg['bf16']}")
    print(f"  Max length:      {max_length}")
    print(f"  Total steps:     ~{total_steps}")
    print(f"  LoRA rank:       {cfg['lora_r']}")
    print(f"  Class weights:   {class_weights}")
    print(f"  Output:          {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        report_to="none",
        dataloader_pin_memory=False,   # Avoid issues with 4-bit models
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",      # Memory-efficient optimizer with QLoRA
        lr_scheduler_type="cosine",
    )

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        ],
    )

    # ── Train ───────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save best model (merge adapters) ───────────────────────────────────
    print(f"\nSaving best LoRA adapters to {BEST_DIR}...")
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    # Save adapter-only (fast, small)
    trainer.model.save_pretrained(str(BEST_DIR))
    tokenizer.save_pretrained(str(BEST_DIR))

    # Also merge and save full model for single-file inference
    merged_dir = OUTPUT_DIR / "merged"
    print(f"Merging adapters into full model → {merged_dir}...")
    try:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"  ✓ Merged model saved to {merged_dir}")
    except Exception as e:
        print(f"  ⚠️  Could not merge (will use adapters): {e}")

    # ── Final evaluation ────────────────────────────────────────────────────
    print("\nFinal evaluation on test set:")
    results = trainer.evaluate(test_tok)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n✓ Training complete. Best model at: {BEST_DIR}")
    return results


if __name__ == "__main__":
    train()
