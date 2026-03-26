#!/usr/bin/env python3
"""
Upload models/detector/llm_detector/best to Hugging Face Hub.

Credentials (never commit):
  HF_TOKEN       — from .env (create at https://huggingface.co/settings/tokens)
  HF_MODEL_REPO  — optional, default umairinayat/llm_detector

Usage (from project root, venv active):
  python scripts/push_llm_detector_to_hf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv(PROJECT_ROOT / ".env")

BEST_DIR = PROJECT_ROOT / "models" / "detector" / "llm_detector" / "best"
README_SRC = PROJECT_ROOT / "model_cards" / "llm_detector_README.md"


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Add it to .env (see https://huggingface.co/settings/tokens)")
        sys.exit(1)

    repo_id = os.getenv("HF_MODEL_REPO", "umairinayat/llm_detector").strip()
    private = os.getenv("HF_PRIVATE_REPO", "0").strip().lower() in ("1", "true", "yes")

    if not BEST_DIR.is_dir():
        print(f"ERROR: missing checkpoint directory: {BEST_DIR}")
        sys.exit(1)

    if not README_SRC.is_file():
        print(f"ERROR: missing README template: {README_SRC}")
        sys.exit(1)

    readme_text = README_SRC.read_text(encoding="utf-8")

    api = HfApi(token=token)
    print(f"Creating repo if needed: {repo_id} (private={private})")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)

    print(f"Uploading adapter + tokenizer from {BEST_DIR} (excluding local README.md)")
    api.upload_folder(
        folder_path=str(BEST_DIR),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Sync best QLoRA checkpoint (adapters + tokenizer)",
        ignore_patterns=["README.md", ".git"],
    )

    print("Uploading model_cards/llm_detector_README.md as README.md")
    api.upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Update model card README",
    )

    print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
