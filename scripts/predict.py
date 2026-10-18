#!/usr/bin/env python3
"""Inference entry point.

Usage:
    python scripts/predict.py --input test.parquet --output predictions.parquet

The script is resilient by design: if no GPU/model is available it still
produces predictions using the deterministic rule engine, so the submission
format is always valid.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make ``src`` importable when the package is not installed (local runs).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.config import RuntimeConfig  # noqa: E402
from guardian.io_utils import dump_debug, read_table, write_predictions  # noqa: E402
from guardian.pipeline import Detector  # noqa: E402


def _coerce(raw: str):
    low = raw.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _assign(target: dict, dotted: str, value) -> None:
    """Set ``a.b.c = value`` inside a nested dict, creating levels as needed."""
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guardian of Truth inference")
    parser.add_argument("--input", required=True, help="path to test parquet/csv/jsonl")
    parser.add_argument("--output", required=True, help="path to predictions file")
    parser.add_argument("--config", default=None, help="JSON config (see configs/default.json)")
    parser.add_argument("--model", default=None, help="HF model id or local path")
    parser.add_argument("--model-path", default=None, help="local directory with weights")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["auto", "vllm", "transformers", "heuristic"],
        help="inference backend (auto: vllm -> transformers -> rules)",
    )
    parser.add_argument("--limit", type=int, default=None, help="predict only the first N rows")
    parser.add_argument("--threshold", type=float, default=None, help="decision threshold")
    parser.add_argument("--rule-mode", default=None, choices=["strict", "hint"])
    parser.add_argument("--time-budget", type=float, default=None, help="seconds, default 1500")
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--self-consistency", type=int, default=None)
    parser.add_argument("--with-details", action="store_true", help="extra diagnostic columns")
    parser.add_argument("--debug-jsonl", default=None, help="dump per-row diagnostics")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key by dotted path, e.g. --set judge.verify_negatives=false",
    )
    parser.add_argument("--dump-config", default=None, help="write the resolved config and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("guardian.predict")

    overrides: dict = {}
    if args.model or args.model_path:
        overrides.setdefault("model", {})
        if args.model:
            overrides["model"]["id"] = args.model
        if args.model_path:
            overrides["model"]["path"] = args.model_path
    if args.max_prompt_tokens:
        overrides.setdefault("model", {})["max_prompt_tokens"] = args.max_prompt_tokens
    if args.threshold is not None:
        overrides.setdefault("fusion", {})["threshold"] = args.threshold
    if args.rule_mode:
        overrides.setdefault("fusion", {})["rule_mode"] = args.rule_mode
    if args.time_budget:
        overrides.setdefault("runtime", {})["time_budget_sec"] = args.time_budget
    if args.self_consistency:
        overrides.setdefault("judge", {})["self_consistency"] = args.self_consistency
    if args.backend:
        overrides["backend"] = args.backend
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        _assign(overrides, key.strip(), _coerce(raw.strip()))

    config = RuntimeConfig.load(args.config, overrides)
    if args.dump_config:
        config.dump(args.dump_config)
        logger.info("Config written to %s", args.dump_config)
        return 0

    started = time.monotonic()
    rows = read_table(args.input)
    if args.limit:
        rows = rows[: args.limit]
    logger.info("Predicting %d rows", len(rows))

    detector = Detector(config, backend_name=args.backend or config.raw.get("backend", "auto"))
    deadline = started + float(config.runtime.get("time_budget_sec", 1500))
    predictions = detector.predict_rows(rows, deadline=deadline)

    write_predictions(args.output, predictions, with_details=args.with_details)

    if args.debug_jsonl:
        debug_records = []
        for row, pred in zip(rows, predictions):
            debug_records.append(
                {
                    "id": pred.id,
                    "label": pred.label,
                    "score": pred.score,
                    "judge_label": pred.judge_label,
                    "judge_confidence": pred.judge_confidence,
                    "reason": pred.reason,
                    "signals": [
                        {"kind": s.kind, "severity": s.severity, "message": s.message}
                        for s in pred.signals
                    ],
                    "prompt_chars": len(str(row.get("prompt") or "")),
                    "response_chars": len(str(row.get("response") or "")),
                }
            )
        dump_debug(args.debug_jsonl, debug_records)

    summary = {
        **detector.stats,
        "positive_rate": round(
            sum(p.label for p in predictions) / max(1, len(predictions)), 4
        ),
        "total_elapsed_sec": round(time.monotonic() - started, 2),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
