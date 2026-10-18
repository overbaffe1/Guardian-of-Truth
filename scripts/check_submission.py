#!/usr/bin/env python3
"""Validate the submission file against the platform contract."""

from __future__ import annotations

import argparse

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="test parquet used for inference")
    parser.add_argument("--preds", required=True, help="predictions file")
    args = parser.parse_args(argv)

    test = pd.read_parquet(args.input) if args.input.endswith((".parquet", ".pq")) else pd.read_csv(args.input)
    preds = pd.read_parquet(args.preds) if args.preds.endswith((".parquet", ".pq")) else pd.read_csv(args.preds)

    problems: list[str] = []
    for column in ("id", "label"):
        if column not in preds.columns:
            problems.append(f"missing column: {column}")
    if problems:
        print("\n".join(problems))
        return 1

    if len(preds) != len(test):
        problems.append(f"row count mismatch: {len(preds)} != {len(test)}")
    if preds["id"].astype(str).duplicated().any():
        problems.append("duplicate ids in predictions")
    if set(preds["label"].dropna().unique()) - {0, 1}:
        problems.append(f"labels must be 0/1, got {sorted(preds['label'].dropna().unique())[:10]}")
    if preds["label"].isna().any():
        problems.append("NaN labels")
    missing = set(test["id"].astype(str)) - set(preds["id"].astype(str))
    extra = set(preds["id"].astype(str)) - set(test["id"].astype(str))
    if missing:
        problems.append(f"{len(missing)} ids from the test set are missing, e.g. {sorted(missing)[:5]}")
    if extra:
        problems.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:5]}")

    if problems:
        print("INVALID:")
        for problem in problems:
            print(" -", problem)
        return 1

    rate = preds["label"].mean()
    print(f"OK: {len(preds)} rows, positive rate {rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
