#!/usr/bin/env python3
"""Download judge weights into ``model/`` (used at Docker build time).

    python scripts/download_model.py --model-id Qwen/Qwen2.5-14B-Instruct-AWQ --out model

Requires network access to Hugging Face.  If the weights are shipped inside
the archive instead, this step can be skipped entirely.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("GUARDIAN_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct-AWQ"))
    parser.add_argument("--out", default=os.environ.get("GUARDIAN_MODEL_PATH", "model"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"huggingface_hub is not installed: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} -> {out}")
    snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(out),
        token=args.token,
        ignore_patterns=["*.gguf", "*.pth", "original/*", "*.h5"],
    )
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
