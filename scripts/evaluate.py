#!/usr/bin/env python3
"""Evaluate the detector on a labelled file (public split or synthetic devset).

    python scripts/evaluate.py --data data/dev_synthetic.jsonl --backend heuristic

For the real public split use the parquet from the competition and run with
the GPU backend (omit ``--backend heuristic``).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.config import RuntimeConfig  # noqa: E402
from guardian.io_utils import compute_metrics, read_table  # noqa: E402
from guardian.pipeline import Detector  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers", "heuristic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--rule-mode", default=None, choices=["strict", "hint"])
    parser.add_argument("--dump", default=None, help="write per-row diagnostics to this jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict = {}
    if args.threshold is not None:
        overrides["fusion"] = {"threshold": args.threshold}
    if args.rule_mode:
        overrides.setdefault("fusion", {})["rule_mode"] = args.rule_mode
    config = RuntimeConfig.load(args.config, overrides)

    rows = read_table(args.data)
    rows = [row for row in rows if row.get("label") is not None]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No labelled rows found", file=sys.stderr)
        return 2

    detector = Detector(config, backend_name=args.backend)
    started = time.monotonic()
    predictions = detector.predict_rows(rows)
    elapsed = time.monotonic() - started

    y_true = [int(row["label"]) for row in rows]
    y_pred = [pred.label for pred in predictions]
    metrics = compute_metrics(y_true, y_pred)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    print(
        json.dumps(
            {
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                "positive_rate": round(sum(y_pred) / len(y_pred), 4),
                "elapsed_sec": round(elapsed, 2),
                "detector": detector.stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Breakdown by mutation type when the synthetic generator was used.
    if any("_raw" in row and isinstance(row["_raw"].get("meta"), dict) for row in rows):
        buckets: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for row, pred in zip(rows, predictions):
            meta = (row.get("_raw") or {}).get("meta") or {}
            key = f"{domain_label(row)}:{meta.get('mutation', 'clean')}" if meta else domain_label(row)
            buckets[key][0] += 1
            buckets[key][1] += int(pred.label == int(row["label"]))
        print("\nby bucket (rows, correct):")
        for key, (n, correct) in sorted(buckets.items()):
            print(f"  {key:45s} {n:4d}  {correct / max(1, n):.3f}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for row, pred in zip(rows, predictions):
                fh.write(
                    json.dumps(
                        {
                            "id": pred.id,
                            "true": int(row["label"]),
                            "pred": pred.label,
                            "score": pred.score,
                            "reason": pred.reason,
                            "signals": [s.kind for s in pred.signals],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"\nWrote {args.dump}")
    return 0


def domain_label(row: dict) -> str:
    raw = row.get("_raw") or {}
    meta = raw.get("meta") if isinstance(raw, dict) else None
    if isinstance(meta, dict) and meta.get("domain"):
        return str(meta["domain"])
    return "all"


if __name__ == "__main__":
    raise SystemExit(main())
