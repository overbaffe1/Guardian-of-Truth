#!/usr/bin/env python3
"""Blend several prediction files into one (majority vote or mean score).

Useful when you have several submissions (different models, thresholds, or
prompt variants) and want a safer ensemble:

    python scripts/blend.py --preds a.parquet b.parquet c.parquet --out merged.parquet

    # mean of calibrated scores instead of hard-vote majority
    python scripts/blend.py --preds a.parquet b.parquet --how score --threshold 0.45

Inputs must contain ``id`` and ``label``; ``score`` is used when it exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.io_utils import read_table, write_predictions  # noqa: E402
from guardian.types import Prediction  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend prediction files")
    parser.add_argument("--preds", nargs="+", required=True, help="two or more prediction files")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--how",
        choices=["vote", "score"],
        default="vote",
        help="vote: majority of labels; score: mean score vs threshold",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for --how score")
    parser.add_argument("--min-votes", type=int, default=None,
                        help="labels needed for 1 in vote mode (default: strict majority)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.preds) < 2:
        raise SystemExit("--preds needs at least two files")

    tables = [read_table(path) for path in args.preds]
    order: list[str] = []
    labels: dict[str, list[int]] = {}
    scores: dict[str, list[float]] = {}
    for index, table in enumerate(tables):
        for row in table:
            uid = str(row.get("id"))
            if uid not in labels:
                if index > 0:
                    raise SystemExit(f"id {uid!r} is missing from {args.preds[0]}")
                order.append(uid)
                labels[uid] = []
                scores[uid] = []
            if uid not in labels:
                continue
            try:
                labels[uid].append(int(row.get("label", 0)))
            except (TypeError, ValueError):
                labels[uid].append(0)
            raw_score = row.get("score")
            if raw_score is not None:
                try:
                    scores[uid].append(float(raw_score))
                except (TypeError, ValueError):
                    pass

    predictions: list[Prediction] = []
    for uid in order:
        votes = labels[uid]
        need = args.min_votes if args.min_votes is not None else len(args.preds) // 2 + 1
        if args.how == "vote":
            positive = sum(1 for v in votes if v == 1)
            label = 1 if positive >= need else 0
            pred_score = positive / max(1, len(votes))
        else:
            mean = sum(scores[uid]) / len(scores[uid]) if scores[uid] else 0.0
            label = 1 if mean >= args.threshold else 0
            pred_score = mean
        predictions.append(
            Prediction(
                id=uid,
                label=label,
                score=pred_score,
                reason=f"blend of {len(args.preds)} files ({args.how})",
                backend="blend",
            )
        )

    write_predictions(args.out, predictions)
    positives = sum(p.label for p in predictions)
    print(
        f"Blended {len(predictions)} rows from {len(args.preds)} files "
        f"({args.how}); positive_rate={positives / max(1, len(predictions)):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
