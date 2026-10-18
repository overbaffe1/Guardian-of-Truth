#!/usr/bin/env python3
"""Build a competition-ready submission archive.

The platform is picky in ways that are easy to get wrong and impossible to
debug from the outside (participants report opaque failures at stage 08
"data unpacking" and stage 22 "metric calculation"):

* **ZIP only.** ``tar`` and ``tar.gz`` archives were reported to fail.
* **No wrapping directory.** The platform runs ``<extracted>/scripts/predict.py``,
  so ``pyproject.toml`` and ``scripts/`` must sit at the root of the archive.
* **No ``metrics/`` directory.** The organisers explicitly asked for it to be
  left out.
* **Readable permissions.** A zip built on Windows (or via ``zipfile``) can
  store directories as ``0600``, so the unpacker running as a normal user cannot
  enter them and stage 08 fails. Directories are stored as ``0755`` and files as
  ``0644``.
* **Parquet predictions.** Handled by ``guardian.io_utils``: the scorer reads
  Parquet even when ``--output`` ends in ``.csv``.
* **Archive under 40 GB.** The limit applies to the uploaded archive.

Usage::

    python scripts/make_submission.py --out submission.zip
    python scripts/make_submission.py --out submission.zip --model /path/to/weights
    python scripts/make_submission.py --check          # validate only
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Never shipped: caches, VCS metadata, local artefacts, anything the platform
#: rejects or that only bloats the archive.
EXCLUDE_DIRS = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".venv", "venv", "build", "dist", "*.egg-info", "metrics", "data", ".hf_cache",
    ".idea", ".vscode", "reports", "tests",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".log", ".parquet", ".csv", ".jsonl"}
EXCLUDE_NAMES = {".DS_Store", ".gitignore", ".dockerignore", "Makefile", "Dockerfile"}

#: The platform runs ``scripts/predict.py`` from the root of the archive.
REQUIRED = ["README.md", "pyproject.toml", "scripts/predict.py", "src"]
MAX_ARCHIVE_BYTES = 40 * 1024**3

DIR_MODE = 0o755
FILE_MODE = 0o644


def _is_excluded(path: Path) -> bool:
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_DIRS or part.endswith(".egg-info"):
            return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    return False


def collect(root: Path, model_dir: Path | None = None) -> list[Path]:
    """Return the files to archive, relative to ``root``."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _is_excluded(rel):
            continue
        if path.is_dir():
            continue
        if model_dir is not None and rel.parts[0] == "model" and not rel.parts[1:]:
            # placeholder kept only when a checkpoint is mounted
            continue
        files.append(rel)
    return files


def check_structure(root: Path) -> list[str]:
    problems: list[str] = []
    for required in REQUIRED:
        if not (root / required).exists():
            problems.append(f"missing required entry: {required}")
    model = root / "model"
    if not model.is_dir():
        problems.append("missing model/ directory")
    else:
        has_weights = any(
            p.suffix in {".safetensors", ".bin", ".gguf", ".pt"} for p in model.rglob("*")
        )
        if not has_weights:
            problems.append(
                "model/ contains no weights (.safetensors/.bin); the scoring "
                "environment provides none, so they must ship in the archive"
            )
    if (root / "metrics").exists():
        problems.append("archive must not contain a metrics/ directory")
    return problems


def build(root: Path, out: Path, model_dir: Path | None = None) -> int:
    files = collect(root, model_dir)
    if model_dir is not None:
        for extra in sorted(model_dir.rglob("*")):
            if extra.is_file():
                files.append(Path("model") / extra.relative_to(model_dir))

    tmp = out.with_suffix(out.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Explicit root entries with sane permissions, including empty dirs the
        # unpacker must be able to enter.
        for name in ("scripts", "src", "model"):
            info = zipfile.ZipInfo(name + "/")
            info.external_attr = (DIR_MODE | 0o40000) << 16 | 0x10
            zf.writestr(info, b"")
        for rel in files:
            source = root / rel
            if model_dir is not None and rel.parts[0] == "model" and not (root / rel).exists():
                continue
            info = zipfile.ZipInfo(str(rel).replace(os.sep, "/"))
            info.external_attr = FILE_MODE << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(source, "rb") as fh:
                zf.writestr(info, fh.read())
    tmp.replace(out)
    return out.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO))
    parser.add_argument("--out", default="submission.zip")
    parser.add_argument("--model", default=None, help="directory of judge weights to place in model/")
    parser.add_argument("--check", action="store_true", help="validate only, do not build")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    problems = check_structure(root)
    for problem in problems:
        print(f"WARNING: {problem}", file=sys.stderr)

    if args.check:
        return 1 if problems else 0

    out = Path(args.out).resolve()
    size = build(root, out, Path(args.model).resolve() if args.model else None)
    print(f"Wrote {out} ({size / 1024**3:.2f} GiB)")
    if size >= MAX_ARCHIVE_BYTES:
        print("ERROR: archive exceeds the 40 GB limit", file=sys.stderr)
        return 1

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    print(f"entries: {len(names)}")
    for required in ("README.md", "pyproject.toml", "scripts/predict.py"):
        status = "ok" if required in names else "MISSING"
        print(f"  {required}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())