#!/usr/bin/env python3
"""Per-signal precision audit on a labelled split (deterministic rules only).

    python scripts/signal_audit.py --data valid.parquet

Reports, for every rule, how often it fires and how often it is right, plus the
"positives that no rule touches" count -- the ceiling of the rule stage and the
workload handed to the LLM judge. Only aggregates are printed, never raw rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.config import RuntimeConfig  # noqa: E402
from guardian.io_utils import read_table  # noqa: E402
from guardian.signals import compute_signals  # noqa: E402
from guardian.toolspec import extract_available_tools  # noqa: E402
from guardian.transcript import parse_transcript, parse_response  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dump-signals", default=None, help="write id/signal/label triples")
    parser.add_argument("--dump-misses", default=None, help="write positives with no signal")
    return parser.parse_args(argv)


def _stage(n: int, tp: int, total_pos: int, total: int) -> dict:
    """Precision/recall/F1 of predicting positive for the rows a stage fires on."""
    if n == 0:
        return {"n": 0, "tp": 0, "precision": None, "recall": 0.0, "f1": 0.0}
    precision = tp / n
    recall = tp / max(1, total_pos)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "tp": tp,
        "fp": n - tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "coverage": round(n / total, 4),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RuntimeConfig.load(args.config)

    rows = [r for r in read_table(args.data) if r.get("label") is not None]
    if args.limit:
        rows = rows[: args.limit]

    per_kind: dict[str, dict] = {}
    by_domain: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "pos": 0, "hard": 0, "hard_tp": 0, "any": 0, "any_tp": 0}
    )
    miss_ids: list[str] = []
    triples: list[str] = []
    any_signal_pos = 0
    hard_n = 0
    hard_tp = 0
    any_n = 0
    total_pos = 0

    for row in rows:
        label = int(row["label"])
        rid = str(row.get("id", ""))
        domain = rid.split("__")[0]
        transcript = parse_transcript(str(row.get("prompt", "")))
        response = parse_response(str(row.get("response", "")))
        specs = extract_available_tools(transcript.system_text)
        signals = compute_signals(transcript, response, specs, config.signals_config())
        kinds = {s.kind for s in signals}
        hard_kinds = {s.kind for s in signals if s.severity == "hard"}

        total_pos += label
        if hard_kinds:
            hard_n += 1
            hard_tp += label
        if kinds:
            any_n += 1
        bucket = by_domain[domain]
        bucket["n"] += 1
        bucket["pos"] += label
        if hard_kinds:
            bucket["hard"] += 1
            bucket["hard_tp"] += label
        if kinds:
            bucket["any"] += 1
            bucket["any_tp"] += label

        for kind in kinds:
            stat = per_kind.setdefault(
                kind,
                {
                    "n": 0,
                    "tp": 0,
                    "hard_n": 0,
                    "hard_tp": 0,
                    "domains": collections.Counter(),
                },
            )
            stat["n"] += 1
            stat["tp"] += label
            stat["domains"][domain] += 1
            if kind in hard_kinds:
                stat["hard_n"] += 1
                stat["hard_tp"] += label

        if label == 1:
            if kinds:
                any_signal_pos += 1
            else:
                miss_ids.append(rid)
        if args.dump_signals:
            triples.append(
                json.dumps(
                    {"id": rid, "label": label, "signals": sorted(kinds)},
                    ensure_ascii=False,
                )
            )

    report: dict = {
        "n": len(rows),
        "positives": total_pos,
        "negatives": len(rows) - total_pos,
        "stage": {
            "hard_only": _stage(hard_n, hard_tp, total_pos, len(rows)),
            "any_signal": _stage(any_n, any_signal_pos, total_pos, len(rows)),
        },
        "signals": {},
        "by_domain": {},
        "positives_without_any_signal": len(miss_ids),
        "positives_with_any_signal": any_signal_pos,
        "positives_without_hard_signal": total_pos - hard_tp,
    }

    for kind, stat in sorted(per_kind.items(), key=lambda kv: -kv[1]["n"]):
        n = stat["n"]
        hn = stat["hard_n"]
        report["signals"][kind] = {
            "n": n,
            "precision": round(stat["tp"] / n, 4) if n else None,
            "recall_of_positives": round(stat["tp"] / max(1, total_pos), 4),
            "coverage": round(n / len(rows), 4),
            "hard": hn > 0,
            "hard_fraction": round(hn / n, 4) if n else None,
            "hard_precision": round(stat["hard_tp"] / hn, 4) if hn else None,
            "domains": dict(stat["domains"]),
        }

    for domain, b in sorted(by_domain.items()):
        report["by_domain"][domain] = {
            **b,
            "hard_precision": round(b["hard_tp"] / b["hard"], 4) if b["hard"] else None,
            "any_precision": round(b["any_tp"] / b["any"], 4) if b["any"] else None,
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.dump_signals:
        Path(args.dump_signals).write_text("\n".join(triples) + "\n", encoding="utf-8")
    if args.dump_misses:
        Path(args.dump_misses).write_text("\n".join(miss_ids) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())