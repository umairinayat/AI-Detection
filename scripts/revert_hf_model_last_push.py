#!/usr/bin/env python3
"""
Restore a Hugging Face model repo to a chosen revision.

  - Default: the parent of the latest commit (undo last Hub push).
  - Or pass --revision <sha> to pin the repo to that exact commit (removes any
    files not in that revision, then re-uploads its snapshot).

Uses HF_TOKEN and optional HF_MODEL_REPO from .env (same as push_llm_detector_to_hf.py).

Usage (from project root):
  python scripts/revert_hf_model_last_push.py
  python scripts/revert_hf_model_last_push.py --revision 2a4d394b116103f881ac291ebf1c15f33e50289d
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi, snapshot_download

load_dotenv(PROJECT_ROOT / ".env")


def _paths_under(root: Path) -> set[str]:
    out: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file():
            out.add(p.relative_to(root).as_posix())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore HF model repo to a revision.")
    parser.add_argument(
        "--revision",
        metavar="SHA",
        help="Full Hub commit SHA to release (default: parent of latest commit)",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in .env")
        sys.exit(1)

    repo_id = os.getenv("HF_MODEL_REPO", "umairinayat/llm_detector").strip()
    api = HfApi(token=token)

    if args.revision:
        target_sha = args.revision.strip()
        print(f"Target revision (explicit): {target_sha}")
    else:
        commits = api.list_repo_commits(repo_id, repo_type="model")
        if len(commits) < 2:
            print(
                f"ERROR: repo {repo_id} has fewer than 2 commits; nothing to revert to.",
                file=sys.stderr,
            )
            sys.exit(1)
        # commits[0] = latest on main; commits[1] = parent (previous model state)
        target_sha = commits[1].commit_id
        latest_title = commits[0].title
        print(f"Latest commit: {commits[0].commit_id[:7]} — {latest_title!r}")
        print(f"Reverting to parent: {target_sha[:7]}")

    prev_sha = target_sha

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(tmp_path),
            revision=prev_sha,
            token=token,
        )
        prev_paths = _paths_under(tmp_path)

        current_files = api.list_repo_files(repo_id, repo_type="model")
        to_delete = sorted(set(current_files) - prev_paths)
        if to_delete:
            print(f"Removing {len(to_delete)} file(s) not in target revision...")
            ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=ops,
                commit_message=f"Remove files not in revision {prev_sha[:7]}",
                token=token,
            )

        print("Uploading snapshot of target revision...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Release model state at revision {prev_sha[:7]}",
        )

    print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
