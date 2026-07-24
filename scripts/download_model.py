#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
DEFAULT_LOCAL_DIR = "./model"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the complete LFM2.5 model snapshot for local GPU testing.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing an optional HF token.")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} into {local_dir.resolve()}")
    if token:
        print(f"Using Hugging Face token from ${args.token_env}; token value is not logged.")
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        token=token,
        local_dir_use_symlinks=False,
    )
    _write_model_gitignore(local_dir)
    verify_model_dir(local_dir)
    total_bytes = directory_size_bytes(local_dir)
    metadata = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot_path": str(snapshot_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "total_size_bytes": total_bytes,
    }
    with (local_dir / ".baseline_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Model snapshot verified. Total size: {format_bytes(total_bytes)}")
    return 0


def verify_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory is missing: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing required model config: {model_dir / 'config.json'}")
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin")) + list(model_dir.glob("*.pt"))
    if not weight_files:
        raise FileNotFoundError("No model weight files found (*.safetensors, *.bin, or *.pt)")
    tokenizer_candidates = [
        model_dir / "tokenizer.json",
        model_dir / "tokenizer.model",
        model_dir / "tokenizer_config.json",
        model_dir / "vocab.json",
    ]
    if not any(path.exists() for path in tokenizer_candidates):
        raise FileNotFoundError("No tokenizer files found; expected tokenizer.json, tokenizer.model, tokenizer_config.json, or vocab.json")


def directory_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _write_model_gitignore(model_dir: Path) -> None:
    gitignore = model_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

