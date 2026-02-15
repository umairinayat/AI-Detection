"""
Dataset preparation for AI text detection training.

Downloads and preprocesses datasets from Hugging Face:
    - Hello-SimpleAI/HC3: Human vs ChatGPT comparison corpus
    - dmitva/human_ai_generated_text: Large-scale human/AI pairs
    - liamdugan/raid: RAID benchmark (11 LLMs + adversarial attacks)
    - artem9k/ai-text-detection-pile: Large AI text detection pile

Also supports:
    - Local LLM generation via ollama (Llama, Mistral)
    - Mixed-content sample generation
    - Adversarial sample generation (paraphrasing)

Outputs a unified format: {"text": str, "label": int}
    label 0 = Human, label 1 = AI
"""

import json
import random
from pathlib import Path
from typing import Optional

from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets,
    ClassLabel,
    Features,
    Value,
)
from huggingface_hub import hf_hub_download

import config

# Output directory for processed data
DATA_DIR = config.DATA_DIR


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_hc3(max_samples: int = 10000) -> Optional[Dataset]:
    """
    Load HC3 (Human ChatGPT Comparison Corpus).
    Source: Hello-SimpleAI/HC3
    Contains: Human answers vs ChatGPT (GPT-3.5) answers
    """
    print("[HC3] Downloading dataset...")

    subsets = ["finance", "medicine", "open_qa", "wiki_csai"]
    texts, labels = [], []

    for subset in subsets:
        try:
            filepath = hf_hub_download(
                repo_id="Hello-SimpleAI/HC3",
                filename=f"{subset}.jsonl",
                repo_type="dataset",
            )
        except Exception:
            try:
                filepath = hf_hub_download(
                    repo_id="Hello-SimpleAI/HC3",
                    filename=f"{subset}.json",
                    repo_type="dataset",
                )
            except Exception as e:
                print(f"  [HC3] Skipping subset '{subset}': {e}")
                continue

        print(f"  [HC3] Loading subset: {subset}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                for answer in row.get("human_answers", []):
                    if isinstance(answer, str) and len(answer.strip()) > 50:
                        texts.append(answer.strip())
                        labels.append(0)

                for answer in row.get("chatgpt_answers", []):
                    if isinstance(answer, str) and len(answer.strip()) > 50:
                        texts.append(answer.strip())
                        labels.append(1)

        if len(texts) >= max_samples * 2:
            break

    if not texts:
        print("[HC3] WARNING: No data loaded. Falling back to direct load...")
        try:
            ds = load_dataset("Hello-SimpleAI/HC3", split="train")
            for row in ds:
                for answer in row.get("human_answers", []):
                    if isinstance(answer, str) and len(answer.strip()) > 50:
                        texts.append(answer.strip())
                        labels.append(0)
                for answer in row.get("chatgpt_answers", []):
                    if isinstance(answer, str) and len(answer.strip()) > 50:
                        texts.append(answer.strip())
                        labels.append(1)
                if len(texts) >= max_samples * 2:
                    break
        except Exception as e:
            print(f"[HC3] ERROR: Could not load dataset: {e}")
            return None

    return _balance_and_create(texts, labels, max_samples, "HC3")


def load_dmitva(max_samples: int = 10000) -> Optional[Dataset]:
    """
    Load dmitva/human_ai_generated_text dataset.
    Structure: {'id', 'human_text', 'ai_text', 'instructions'}
    """
    print("[dmitva] Downloading dataset...")
    try:
        ds = load_dataset("dmitva/human_ai_generated_text", split="train")
    except Exception as e:
        print(f"[dmitva] ERROR: {e}")
        return None

    col_names = ds.column_names
    print(f"[dmitva] Columns: {col_names}")

    texts, labels = [], []
    for row in ds:
        h_text = row.get("human_text")
        if isinstance(h_text, str) and len(h_text.strip()) > 50:
            texts.append(h_text.strip())
            labels.append(0)

        a_text = row.get("ai_text")
        if isinstance(a_text, str) and len(a_text.strip()) > 50:
            texts.append(a_text.strip())
            labels.append(1)

        if len(texts) >= max_samples * 2:
            break

    return _balance_and_create(texts, labels, max_samples, "dmitva")


def load_raid(max_samples: int = 50000) -> Optional[Dataset]:
    """
    Load RAID benchmark dataset.
    Source: liamdugan/raid

    Contains text from 11 different LLMs with adversarial attacks.
    This is the benchmark GPTZero was evaluated on.
    """
    print("[RAID] Downloading dataset...")
    try:
        # RAID has multiple splits and configurations
        ds = load_dataset("liamdugan/raid", split="train")
    except Exception:
        try:
            # Try alternative loading
            ds = load_dataset("liamdugan/raid", "generation", split="train")
        except Exception as e:
            print(f"[RAID] ERROR: Could not load dataset: {e}")
            print("[RAID] Try: pip install datasets --upgrade")
            return None

    col_names = ds.column_names
    print(f"[RAID] Columns: {col_names}")
    print(f"[RAID] Total rows: {len(ds)}")

    texts, labels = [], []

    for row in ds:
        text = row.get("generation") or row.get("text") or row.get("output", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        # Determine label from the dataset
        # RAID typically has a 'model' column: "human" or model name
        model = row.get("model", "")
        if model == "human" or row.get("label", -1) == 0:
            labels.append(0)
        else:
            labels.append(1)
        texts.append(text.strip())

        if len(texts) >= max_samples * 2:
            break

    return _balance_and_create(texts, labels, max_samples, "RAID")


def load_ai_text_pile(max_samples: int = 50000) -> Optional[Dataset]:
    """
    Load artem9k/ai-text-detection-pile dataset.
    Large-scale collection of human vs AI-generated texts.
    """
    print("[AI-Pile] Downloading dataset...")
    try:
        ds = load_dataset("artem9k/ai-text-detection-pile", split="train")
    except Exception as e:
        print(f"[AI-Pile] ERROR: {e}")
        return None

    col_names = ds.column_names
    print(f"[AI-Pile] Columns: {col_names}")
    print(f"[AI-Pile] Total rows: {len(ds)}")

    texts, labels = [], []

    for row in ds:
        text = row.get("text", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        # Determine label
        label = row.get("label", row.get("generated", None))
        if label is None:
            source = row.get("source", "").lower()
            if "human" in source:
                label = 0
            elif any(kw in source for kw in ["gpt", "ai", "llm", "generated"]):
                label = 1
            else:
                continue

        if isinstance(label, str):
            label = 1 if label.lower() in ["ai", "generated", "1", "true"] else 0

        labels.append(int(label))
        texts.append(text.strip())

        if len(texts) >= max_samples * 2:
            break

    return _balance_and_create(texts, labels, max_samples, "AI-Pile")


def load_gptwiki(max_samples: int = 20000) -> Optional[Dataset]:
    """
    Load GPT-wiki-intro dataset.
    Source: aadityaubhat/GPT-wiki-intro
    Contains: Wikipedia intros vs GPT-3.5 rephrasings
    """
    print("[GPT-Wiki] Downloading dataset...")
    try:
        ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    except Exception as e:
        print(f"[GPT-Wiki] ERROR: {e}")
        return None

    col_names = ds.column_names
    print(f"[GPT-Wiki] Columns: {col_names}")

    texts, labels = [], []

    for row in ds:
        # Human text (Wikipedia intro)
        wiki_text = row.get("wiki_intro", "")
        if isinstance(wiki_text, str) and len(wiki_text.strip()) > 50:
            texts.append(wiki_text.strip())
            labels.append(0)

        # AI text (GPT rephrasing)
        gpt_text = row.get("generated_intro", "")
        if isinstance(gpt_text, str) and len(gpt_text.strip()) > 50:
            texts.append(gpt_text.strip())
            labels.append(1)

        if len(texts) >= max_samples * 2:
            break

    return _balance_and_create(texts, labels, max_samples, "GPT-Wiki")


# ============================================================================
# Mixed Content Generation
# ============================================================================

def generate_mixed_samples(
    human_texts: list[str],
    ai_texts: list[str],
    num_samples: int = 5000,
) -> Optional[Dataset]:
    """
    Generate mixed-content samples by combining human and AI paragraphs.
    This helps the model learn to detect partially AI-written documents.

    Note: These are labeled as AI (1) since they contain AI content.
    """
    print(f"[Mixed] Generating {num_samples} mixed-content samples...")

    if len(human_texts) < 100 or len(ai_texts) < 100:
        print("[Mixed] Not enough source texts for mixed generation")
        return None

    mixed_texts = []
    random.seed(42)

    for _ in range(num_samples):
        # Randomly decide mix ratio (30-70% AI)
        num_human = random.randint(1, 3)
        num_ai = random.randint(1, 3)

        paragraphs = []

        # Add human paragraphs
        for _ in range(num_human):
            h_text = random.choice(human_texts)
            # Take a paragraph-sized chunk
            sentences = h_text.split(". ")
            chunk_size = random.randint(1, min(3, len(sentences)))
            paragraphs.append(". ".join(sentences[:chunk_size]) + ".")

        # Add AI paragraphs
        for _ in range(num_ai):
            a_text = random.choice(ai_texts)
            sentences = a_text.split(". ")
            chunk_size = random.randint(1, min(3, len(sentences)))
            paragraphs.append(". ".join(sentences[:chunk_size]) + ".")

        # Shuffle paragraph order
        random.shuffle(paragraphs)
        mixed_text = " ".join(paragraphs)

        if len(mixed_text.strip()) > 100:
            mixed_texts.append(mixed_text.strip())

    if not mixed_texts:
        return None

    print(f"[Mixed] Generated {len(mixed_texts)} mixed samples")

    return Dataset.from_dict({
        "text": mixed_texts,
        "label": [1] * len(mixed_texts),  # Mixed content → AI label
    })


# ============================================================================
# Helpers
# ============================================================================

def _balance_and_create(
    texts: list[str],
    labels: list[int],
    max_samples: int,
    source_name: str,
) -> Optional[Dataset]:
    """Balance classes and create a Dataset."""
    if not texts:
        print(f"[{source_name}] WARNING: No data loaded.")
        return None

    human = [(t, l) for t, l in zip(texts, labels) if l == 0][:max_samples]
    ai = [(t, l) for t, l in zip(texts, labels) if l == 1][:max_samples]
    min_count = min(len(human), len(ai))

    if min_count == 0:
        print(f"[{source_name}] WARNING: No balanced data available "
              f"(human={len(human)}, ai={len(ai)})")
        return None

    balanced = human[:min_count] + ai[:min_count]

    print(f"[{source_name}] Loaded {min_count} human + {min_count} AI = {min_count * 2} total")
    return Dataset.from_dict({
        "text": [t for t, _ in balanced],
        "label": [l for _, l in balanced],
    })


# ============================================================================
# Main Pipeline
# ============================================================================

def prepare_combined_dataset(
    test_split: float = 0.15,
    include_raid: bool = True,
    include_ai_pile: bool = True,
    include_gptwiki: bool = True,
    include_mixed: bool = True,
) -> dict:
    """
    Download all datasets and combine into a single train/test split.

    Args:
        test_split: Fraction of data for test set
        include_raid: Whether to download RAID benchmark
        include_ai_pile: Whether to download AI text pile
        include_gptwiki: Whether to download GPT-wiki-intro
        include_mixed: Whether to generate mixed-content samples

    Returns:
        dict with "train" and "test" Dataset objects.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets_list = []
    all_human_texts = []
    all_ai_texts = []

    # --- Core datasets (always loaded) ---

    # HC3 — Human vs ChatGPT
    hc3_cfg = config.DATA_SOURCES.get("hc3", {})
    hc3 = load_hc3(max_samples=hc3_cfg.get("max_samples", 10000))
    if hc3 is not None:
        datasets_list.append(hc3)
        _collect_texts(hc3, all_human_texts, all_ai_texts)

    # dmitva — Human/AI pairs
    dmitva_cfg = config.DATA_SOURCES.get("dmitva", {})
    dmitva = load_dmitva(max_samples=dmitva_cfg.get("max_samples", 10000))
    if dmitva is not None:
        datasets_list.append(dmitva)
        _collect_texts(dmitva, all_human_texts, all_ai_texts)

    # --- Extended datasets ---

    # RAID — 11 LLMs + adversarial attacks
    if include_raid:
        raid_cfg = config.DATA_SOURCES.get("raid", {})
        raid = load_raid(max_samples=raid_cfg.get("max_samples", 50000))
        if raid is not None:
            datasets_list.append(raid)
            _collect_texts(raid, all_human_texts, all_ai_texts)

    # AI Text Detection Pile
    if include_ai_pile:
        pile_cfg = config.DATA_SOURCES.get("ai_text_pile", {})
        pile = load_ai_text_pile(max_samples=pile_cfg.get("max_samples", 50000))
        if pile is not None:
            datasets_list.append(pile)
            _collect_texts(pile, all_human_texts, all_ai_texts)

    # GPT-Wiki-Intro
    if include_gptwiki:
        gptwiki = load_gptwiki(max_samples=20000)
        if gptwiki is not None:
            datasets_list.append(gptwiki)
            _collect_texts(gptwiki, all_human_texts, all_ai_texts)

    # --- Mixed content generation ---
    if include_mixed and all_human_texts and all_ai_texts:
        mixed = generate_mixed_samples(all_human_texts, all_ai_texts, num_samples=5000)
        if mixed is not None:
            datasets_list.append(mixed)

    if not datasets_list:
        print("ERROR: No datasets could be loaded. Check your internet connection.")
        return None

    # Combine
    combined = concatenate_datasets(datasets_list)
    combined = combined.shuffle(seed=42)

    # Convert label column to ClassLabel for stratification
    features = combined.features.copy()
    features["label"] = ClassLabel(num_classes=2, names=["Human", "AI"])
    combined = combined.cast(features)

    # Train/test split
    split = combined.train_test_split(
        test_size=test_split,
        seed=42,
        stratify_by_column="label",
    )

    # Save to disk
    split["train"].save_to_disk(str(DATA_DIR / "train"))
    split["test"].save_to_disk(str(DATA_DIR / "test"))

    # Stats
    train_labels = split["train"]["label"]
    test_labels = split["test"]["label"]
    stats = {
        "train_total": len(train_labels),
        "train_human": train_labels.count(0),
        "train_ai": train_labels.count(1),
        "test_total": len(test_labels),
        "test_human": test_labels.count(0),
        "test_ai": test_labels.count(1),
        "sources": [ds.__class__.__name__ for ds in datasets_list],
        "num_sources": len(datasets_list),
    }
    print(f"\n{'='*60}")
    print(f"Combined dataset prepared:")
    print(f"  Train: {stats['train_total']} ({stats['train_human']} human, {stats['train_ai']} AI)")
    print(f"  Test:  {stats['test_total']} ({stats['test_human']} human, {stats['test_ai']} AI)")
    print(f"  Sources: {stats['num_sources']}")
    print(f"  Saved to: {DATA_DIR}")
    print(f"{'='*60}")

    # Save stats
    with open(DATA_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return split


def _collect_texts(
    ds: Dataset,
    human_texts: list[str],
    ai_texts: list[str],
):
    """Collect human and AI texts from a dataset for mixed-content generation."""
    for row in ds:
        text = row["text"]
        if row["label"] == 0:
            human_texts.append(text)
        else:
            ai_texts.append(text)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare AI detection training data")
    parser.add_argument("--no-raid", action="store_true", help="Skip RAID dataset")
    parser.add_argument("--no-pile", action="store_true", help="Skip AI text pile")
    parser.add_argument("--no-gptwiki", action="store_true", help="Skip GPT-Wiki")
    parser.add_argument("--no-mixed", action="store_true", help="Skip mixed-content generation")
    parser.add_argument("--minimal", action="store_true", help="Only load HC3 + dmitva (fast)")
    args = parser.parse_args()

    if args.minimal:
        prepare_combined_dataset(
            include_raid=False,
            include_ai_pile=False,
            include_gptwiki=False,
            include_mixed=False,
        )
    else:
        prepare_combined_dataset(
            include_raid=not args.no_raid,
            include_ai_pile=not args.no_pile,
            include_gptwiki=not args.no_gptwiki,
            include_mixed=not args.no_mixed,
        )
