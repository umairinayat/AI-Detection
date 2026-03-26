"""
Sentence-level dataset builder for AI text detection.

Datasets (exact filters as specified):
  liamdugan/raid                        model == "human"  → Human (0)
                                        model != "human"  → AI    (1)
  silentone0725/ai-human-text-detection-v1  label == 0    → Human (0)
                                            label == 1    → AI    (1)
  dmitva/human_ai_generated_text        source == "human" → Human (0)
                                        source != "human" → AI    (1)
  NabeelShar/ai_and_human_text          Label == 0        → Human (0)
                                        Label == 1        → AI    (1)
  badhanr/wikipedia_human_written_text  all rows          → Human (0)

Quality filters per sentence:
  8 <= word count <= 80
  No duplicate sentences (md5 dedup)

Usage:
    python scripts/prepare_sentence_data.py
    # ~5 lakh total sentences (500k): 250k per class
    python scripts/prepare_sentence_data.py --target 250000
"""

import argparse
import hashlib
import logging
import random
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset, load_dataset

from detector.preprocessor import split_sentences

# ── logging: same file as training so the full run is in one log ──────────────
LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "train_under_all.log"

def _setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)

_setup_logging()
log = logging.getLogger(__name__)
log.info(
    "\n%s\n[prepare_sentence_data] RUN START %s\n%s",
    "=" * 60,
    datetime.now().isoformat(timespec="seconds"),
    "=" * 60,
)
log.info(f"Log file (append): {LOG_FILE}")


# ── helpers ──────────────────────────────────────────────────────────────────

def _fp(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _ok(sent: str) -> bool:
    words = sent.split()
    if not (8 <= len(words) <= 80):
        return False
    non_ascii = sum(1 for c in sent if ord(c) > 127)
    if non_ascii / max(len(sent), 1) > 0.4:
        return False
    return True


def _add(text: str, label: int, bucket: list, seen: set, cap: int) -> int:
    """Split text into sentences and add qualifying ones to bucket."""
    if len(bucket) >= cap or not text:
        return 0
    added = 0
    for sent in split_sentences(text):
        if len(bucket) >= cap:
            break
        if not _ok(sent):
            continue
        fp = _fp(sent)
        if fp in seen:
            continue
        seen.add(fp)
        bucket.append({"text": sent, "label": label})
        added += 1
    return added


def _progress(name: str, ai: list, human: list):
    log.info(f"  [{name}] AI={len(ai):,}  Human={len(human):,}")


# ── per-dataset loaders ───────────────────────────────────────────────────────

def load_raid(ai, human, ai_seen, human_seen, cap):
    """
    liamdugan/raid
      model == "human"  → Human
      model != "human"  → AI
    """
    name = "liamdugan/raid"
    log.info(f"Loading {name} ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for row in ds:
            if len(ai) >= cap and len(human) >= cap:
                break
            text  = row.get("generation") or row.get("text") or ""
            model = str(row.get("model", "")).strip().lower()
            if not text:
                continue
            if model == "human":
                if len(human) < cap:
                    _add(text, 0, human, human_seen, cap)
            else:
                if len(ai) < cap:
                    _add(text, 1, ai, ai_seen, cap)
        _progress(name, ai, human)
    except Exception as e:
        log.warning(f"  [{name}] SKIP: {e}")


def load_silentone(ai, human, ai_seen, human_seen, cap):
    """
    silentone0725/ai-human-text-detection-v1
      label == 0 → Human
      label == 1 → AI
    """
    name = "silentone0725/ai-human-text-detection-v1"
    log.info(f"Loading {name} ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for row in ds:
            if len(ai) >= cap and len(human) >= cap:
                break
            # try common text column names
            text = (row.get("text") or row.get("content") or
                    row.get("sentence") or row.get("essay") or "")
            try:
                lbl = int(row.get("label", -1))
            except Exception:
                continue
            if lbl == 0 and len(human) < cap:
                _add(text, 0, human, human_seen, cap)
            elif lbl == 1 and len(ai) < cap:
                _add(text, 1, ai, ai_seen, cap)
        _progress(name, ai, human)
    except Exception as e:
        log.warning(f"  [{name}] SKIP: {e}")


def load_dmitva(ai, human, ai_seen, human_seen, cap):
    """
    dmitva/human_ai_generated_text
      source == "human" → Human
      source != "human" → AI
    """
    name = "dmitva/human_ai_generated_text"
    log.info(f"Loading {name} ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for row in ds:
            if len(ai) >= cap and len(human) >= cap:
                break
            text   = row.get("text") or ""
            source = str(row.get("source", "")).strip().lower()
            if not text:
                continue
            if source == "human":
                if len(human) < cap:
                    _add(text, 0, human, human_seen, cap)
            else:
                if len(ai) < cap:
                    _add(text, 1, ai, ai_seen, cap)
        _progress(name, ai, human)
    except Exception as e:
        log.warning(f"  [{name}] SKIP: {e}")


def load_nabeel(ai, human, ai_seen, human_seen, cap):
    """
    NabeelShar/ai_and_human_text
      Label == 0 → Human
      Label == 1 → AI
    """
    name = "NabeelShar/ai_and_human_text"
    log.info(f"Loading {name} ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for row in ds:
            if len(ai) >= cap and len(human) >= cap:
                break
            text = row.get("text") or row.get("Text") or row.get("content") or ""
            try:
                lbl = int(row.get("Label", row.get("label", -1)))
            except Exception:
                continue
            if lbl == 0 and len(human) < cap:
                _add(text, 0, human, human_seen, cap)
            elif lbl == 1 and len(ai) < cap:
                _add(text, 1, ai, ai_seen, cap)
        _progress(name, ai, human)
    except Exception as e:
        log.warning(f"  [{name}] SKIP: {e}")


def load_wikipedia_human(human, human_seen, cap):
    """
    badhanr/wikipedia_human_written_text
      All rows → Human (pure human dataset)
    """
    name = "badhanr/wikipedia_human_written_text"
    log.info(f"Loading {name} (human only) ...")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for row in ds:
            if len(human) >= cap:
                break
            text = (row.get("text") or row.get("content") or
                    row.get("article") or row.get("passage") or "")
            if text:
                _add(text, 0, human, human_seen, cap)
        log.info(f"  [{name}] Human={len(human):,}")
    except Exception as e:
        log.warning(f"  [{name}] SKIP: {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(target_per_class: int = 250_000, seed: int = 42):
    random.seed(seed)

    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-source cap: each source contributes at most this many sentences per class.
    # With 5 sources and target=250k, cap=80k gives room for sources with fewer samples
    # while ensuring no single source dominates.
    PER_SOURCE_CAP = 80_000

    log.info("=" * 60)
    log.info(f"Target: {target_per_class:,} per class  →  {target_per_class * 2:,} total")
    log.info(f"Per-source cap: {PER_SOURCE_CAP:,} per class  (ensures diversity)")
    log.info("=" * 60)

    # Collect separately per source so no single source fills the bucket
    all_ai:    list[dict] = []
    all_human: list[dict] = []

    sources = [
        ("raid",       load_raid),
        ("silentone",  load_silentone),
        ("dmitva",     load_dmitva),
        ("nabeel",     load_nabeel),
    ]

    for src_name, loader in sources:
        ai_buf:    list[dict] = []
        human_buf: list[dict] = []
        ai_seen:    set = set()
        human_seen: set = set()

        loader(ai_buf, human_buf, ai_seen, human_seen, PER_SOURCE_CAP)

        log.info(f"  {src_name:12s}  AI={len(ai_buf):>7,}   Human={len(human_buf):>7,}")
        all_ai.extend(ai_buf)
        all_human.extend(human_buf)

    # Wikipedia is human-only — fill remaining human gap
    wiki_buf:  list[dict] = []
    wiki_seen: set = set()
    load_wikipedia_human(wiki_buf, wiki_seen, PER_SOURCE_CAP)
    log.info(f"  {'wikipedia':12s}  AI={0:>7,}   Human={len(wiki_buf):>7,}")
    all_human.extend(wiki_buf)

    # ── shuffle each class pool then sample target_per_class from each ────────
    random.shuffle(all_ai)
    random.shuffle(all_human)

    log.info("")
    log.info(f"Total collected before sampling →  AI: {len(all_ai):,}   Human: {len(all_human):,}")

    ai_final    = all_ai[:target_per_class]
    human_final = all_human[:target_per_class]
    actual      = min(len(ai_final), len(human_final))

    ai_final    = ai_final[:actual]
    human_final = human_final[:actual]

    all_samples = ai_final + human_final
    random.shuffle(all_samples)

    # Log per-source breakdown of final sample
    src_counts: dict = Counter(
        s["text"][:20] for s in all_samples   # rough proxy — actual breakdown below
    )
    dist = Counter(s["label"] for s in all_samples)

    log.info(f"Final balanced dataset:")
    log.info(f"  Total   : {len(all_samples):,}")
    log.info(f"  AI  (1) : {dist[1]:,}")
    log.info(f"  Human(0): {dist[0]:,}")

    split_idx = int(0.9 * len(all_samples))
    train_s = all_samples[:split_idx]
    test_s  = all_samples[split_idx:]

    log.info(f"  Train   : {len(train_s):,}")
    log.info(f"  Test    : {len(test_s):,}")

    Dataset.from_list(train_s).save_to_disk(str(output_dir / "train"))
    Dataset.from_list(test_s).save_to_disk(str(output_dir / "test"))

    log.info("")
    log.info(f"Saved to {output_dir}/train  and  {output_dir}/test")
    log.info("Next: python -m training.train_classifier  (or scripts/run_sentence_pipeline.sh)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=250_000,
                        help="Sentences per class (250000 → 500000 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.target, args.seed)
