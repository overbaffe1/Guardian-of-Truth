#!/usr/bin/env python3
"""Calibrate the decision threshold on the public labelled split.

Typical workflow on the competition machine:

    python scripts/predict.py \
        --input public.parquet --output /tmp/public_preds.parquet --with-details

    python scripts/calibrate.py \
        --preds /tmp/public_preds.parquet --truth public.parquet \
        --out configs/calibrated.json

The script reports:
  * F1 / precision / recall at the default and at the best threshold;
  * the same metrics for the LLM verdict alone (no rules);
  * per-signal diagnostics (how often each rule fires and how precise it is),
    which helps to decide whether a rule should be ``hard`` or ``soft``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from guardian.io_utils import best_threshold, compute_metrics, read_table  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="predictions with --with-details")
    parser.add_argument("--truth", required=True, help="labelled dataset (id, label)")
    parser.add_argument("--out", default="configs/calibrated.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="current threshold")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preds = pd.read_parquet(args.preds) if args.preds.endswith((".parquet", ".pq")) else pd.read_csv(args.preds)
    truth = pd.DataFrame(read_table(args.truth))

    truth = truth[["id", "label"]].rename(columns={"label": "y"})
    truth["id"] = truth["id"].astype(str)
    preds["id"] = preds["id"].astype(str)
    merged = preds.merge(truth, on="id", how="inner")
    if merged.empty:
        print("No overlapping ids between predictions and truth", file=sys.stderr)
        return 2

    y = merged["y"].astype(int).tolist()
    scores = merged["score"].astype(float).tolist()

    report: dict = {"n": len(merged), "positives": int(sum(y))}

    current = [1 if s >= args.threshold else 0 for s in scores]
    report["at_threshold"] = {"threshold": args.threshold, **compute_metrics(y, current)}

    best_t, best_metrics = best_threshold(y, scores)
    report["best_threshold"] = {"threshold": best_t, **best_metrics}

    if "judge_label" in merged:
        judge = merged["judge_label"].fillna(0).astype(int).tolist()
        report["judge_only"] = compute_metrics(y, judge)
        # Simulate a judge threshold using its confidence.
        if "judge_confidence" in merged:
            judge_scores = merged.apply(
                lambda r: r["judge_confidence"] if r["judge_label"] == 1 else 1 - r["judge_confidence"],
                axis=1,
            ).astype(float).tolist()
            t2, m2 = best_threshold(y, judge_scores)
            report["judge_best_threshold"] = {"threshold": t2, **m2}

    # Per-signal diagnostics.
    signal_stats: dict[str, dict] = {}
    if "signals" in merged:
        for kind in sorted({k for row in merged["signals"].fillna("") for k in str(row).split(";") if k.strip()}):
            mask = merged["signals"].fillna("").astype(str).str.contains(kind, regex=False)
            if mask.sum() == 0:
                continue
            tp = int(((merged.loc[mask, "y"] == 1)).sum())
            n = int(mask.sum())
            signal_stats[kind] = {
                "n": n,
                "precision": round(tp / n, 4),
                "coverage": round(n / len(merged), 4),
                "recall_of_positives": round(tp / max(1, sum(y)), 4),
            }
    report["signals"] = signal_stats

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"fusion": {"threshold": best_t}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}. Use it with:  python scripts/predict.py ... --config {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
