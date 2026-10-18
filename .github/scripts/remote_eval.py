#!/usr/bin/env python3
"""Inspect a downloaded dataset/archive on a GitHub runner and report metrics.

The workflow downloads an arbitrary URL (Sber OBS presigned links, HuggingFace,
anything reachable from the Actions network) and then runs this script.  It is
deliberately *aggregate only*: it prints counts, distributions and metrics, and
never writes raw rows — competition data must not be republished.

Usage (inside the workflow)::

    python .github/scripts/remote_eval.py --input /tmp/data.parquet \
        --label public --out reports/public.json

Supported inputs: .parquet/.pq, .csv/.tsv, .jsonl/.ndjson, .json and .zip
(the first tabular file inside the archive is evaluated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TABLE_SUFFIXES = {".parquet", ".pq", ".csv", ".tsv", ".jsonl", ".ndjson", ".json"}


# --------------------------------------------------------------------------- io
def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inspect_zip(path: Path, extract_dir: Optional[Path]) -> dict[str, Any]:
    info: dict[str, Any] = {"entries": [], "tables": []}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        info["n_entries"] = len(entries)
        info["total_uncompressed"] = sum(e.file_size for e in entries)
        info["entries"] = [
            {"name": e.filename, "bytes": e.file_size}
            for e in sorted(entries, key=lambda e: -e.file_size)[:40]
        ]
        info["tables"] = [
            e.filename for e in entries if Path(e.filename).suffix.lower() in TABLE_SUFFIXES
        ]
        if extract_dir is not None and info["tables"]:
            extract_dir.mkdir(parents=True, exist_ok=True)
            for name in info["tables"][:3]:
                archive.extract(name, extract_dir)
    return info


# ------------------------------------------------------------------------ stats
def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def q(frac: float) -> int:
        idx = min(len(ordered) - 1, max(0, int(round(frac * (len(ordered) - 1)))))
        return ordered[idx]
    return {
        "median": q(0.5),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from guardian.toolspec import extract_available_tools
    from guardian.transcript import parse_response, parse_transcript

    prompts = [str(r.get("prompt") or "") for r in rows]
    responses = [str(r.get("response") or "") for r in rows]
    labels = [r.get("label") for r in rows]
    has_labels = all(isinstance(v, (int, float)) and v == v for v in labels)

    domains: dict[str, int] = {}
    with_policy = with_tools = with_calls = text_only = 0
    n_tools: list[int] = []
    for prompt, response in zip(prompts, responses):
        transcript = parse_transcript(prompt)
        domain = transcript.domain
        domains[domain] = domains.get(domain, 0) + 1
        if "<policy>" in transcript.system_text:
            with_policy += 1
        specs = extract_available_tools(transcript.system_text)
        if specs:
            with_tools += 1
            n_tools.append(len(specs))
        turn = parse_response(response)
        if turn.tool_calls:
            with_calls += 1
        else:
            text_only += 1

    stats: dict[str, Any] = {
        "rows": len(rows),
        "prompt_chars": _quantiles([len(p) for p in prompts]),
        "response_chars": _quantiles([len(r) for r in responses]),
        "domains": dict(sorted(domains.items(), key=lambda kv: -kv[1])),
        "with_policy_tag": with_policy,
        "with_tool_catalogue": with_tools,
        "median_tools": sorted(n_tools)[len(n_tools) // 2] if n_tools else 0,
        "responses_with_tool_calls": with_calls,
        "responses_text_only": text_only,
    }
    if has_labels:
        positives = sum(1 for v in labels if int(v) == 1)
        stats["labels"] = {
            "positives": positives,
            "negatives": len(labels) - positives,
            "positive_rate": round(positives / max(1, len(labels)), 4),
        }
        per_domain: dict[str, dict[str, int]] = {}
        for domain, label in zip(domains, labels):
            bucket = per_domain.setdefault(domain, {"n": 0, "pos": 0})
            bucket["n"] += 1
            bucket["pos"] += int(label)
        stats["labels_by_domain"] = per_domain
    return stats


# --------------------------------------------------------------------- signals
def signal_report(
    rows: list[dict[str, Any]], use_labels: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from guardian.config import RuntimeConfig
    from guardian.pipeline import Detector
    from guardian.signals import compute_signals
    from guardian.toolspec import extract_available_tools
    from guardian.transcript import parse_response, parse_transcript

    per_kind: dict[str, dict[str, float]] = {}
    hard_counts = {"rows_with_hard": 0, "hard_true_positive": 0}
    for row in rows:
        transcript = parse_transcript(str(row.get("prompt") or ""))
        response = parse_response(str(row.get("response") or ""))
        specs = extract_available_tools(transcript.system_text)
        signals = compute_signals(transcript, response, specs)
        label = row.get("label")
        positive = isinstance(label, (int, float)) and label == 1
        has_hard = any(s.severity == "hard" for s in signals)
        if has_hard:
            hard_counts["rows_with_hard"] += 1
            hard_counts["hard_true_positive"] += int(positive)
        for kind in {s.kind for s in signals}:
            bucket = per_kind.setdefault(kind, {"n": 0, "pos": 0, "hard": 0})
            bucket["n"] += 1
            bucket["pos"] += int(positive)
            bucket["hard"] += int(any(s.kind == kind and s.severity == "hard" for s in signals))

    positives = sum(1 for r in rows if r.get("label") == 1)
    for kind, bucket in per_kind.items():
        bucket["precision"] = round(bucket["pos"] / bucket["n"], 3)
        bucket["coverage"] = round(bucket["n"] / max(1, len(rows)), 3)
        bucket["recall_of_positives"] = round(bucket["pos"] / max(1, positives), 3)
        bucket["hard_fraction"] = round(bucket["hard"] / bucket["n"], 2)

    detector = Detector(RuntimeConfig.load(), backend=None, auto_load_backend=False)
    predictions = detector.predict_rows(rows)
    from guardian.io_utils import compute_metrics

    metrics: dict[str, Any] = {"rows": len(predictions),
                              "positive_rate": round(sum(p.label for p in predictions) / max(1, len(predictions)), 3)}
    if use_labels:
        y_true = [int(r.get("label") or 0) for r in rows]
        y_pred = [p.label for p in predictions]
        metrics.update(compute_metrics(y_true, y_pred))
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = len(y_true) - tp - fp - fn
        metrics["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        metrics["strict_hard_only"] = {
            "flagged": hard_counts["rows_with_hard"],
            "tp": hard_counts["hard_true_positive"],
            "precision": round(hard_counts["hard_true_positive"] / max(1, hard_counts["rows_with_hard"]), 3),
            "recall": round(hard_counts["hard_true_positive"] / max(1, positives), 3),
        }
        from guardian.io_utils import best_threshold

        threshold, best = best_threshold(y_true, [p.score for p in predictions])
        metrics["best_threshold"] = {"threshold": threshold, **best}

    return per_kind, metrics, hard_counts


# ------------------------------------------------------------------------ main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="downloaded file")
    parser.add_argument("--label", default="dataset", help="tag used in the report")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--limit", type=int, default=0, help="evaluate at most N rows")
    parser.add_argument("--extract-dir", default=None, help="extract zip tables here")
    parser.add_argument("--no-signals", action="store_true", help="skip the rule engine")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"input not found: {path}")

    report: dict[str, Any] = {"label": args.label, "file": file_fingerprint(path)}
    target = path

    if path.suffix.lower() == ".zip":
        extract_dir = Path(args.extract_dir) if args.extract_dir else None
        report["archive"] = inspect_zip(path, extract_dir)
        tables = report["archive"].get("tables") or []
        if extract_dir is not None and tables:
            target = extract_dir / tables[0]
            report["table_used"] = tables[0]
            report["file"] = {**report["file"], **file_fingerprint(target)}

    if target.suffix.lower() in TABLE_SUFFIXES:
        from guardian.io_utils import read_table

        rows = read_table(str(target))
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        report["dataset"] = dataset_stats(rows)
        has_labels = all(isinstance(r.get("label"), (int, float)) for r in rows)
        if not args.no_signals:
            signals, metrics, _ = signal_report(rows, has_labels)
            report["signals"] = signals
            report["heuristic"] = metrics
    else:
        report["note"] = "downloaded file is not a table; only metadata was collected"

    report["elapsed_sec"] = round(time.monotonic() - started, 1)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nreport written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
