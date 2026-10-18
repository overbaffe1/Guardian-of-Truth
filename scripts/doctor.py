#!/usr/bin/env python3
"""Pre-flight check for the competition machine.

Runs a short end-to-end pass and prints everything that can silently break a
submission: missing packages, absent GPU, unresolvable model weights, backend
fallback to rules, and the projected runtime for the full private split.

    python scripts/doctor.py --input public.parquet --limit 8

Exit code is 0 when the pipeline produced predictions (even in rule mode) and
1 only when it could not run at all, so it can be used in CI-like checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.config import RuntimeConfig  # noqa: E402
from guardian.io_utils import read_table  # noqa: E402
from guardian.pipeline import Detector  # noqa: E402

PRIVATE_ROWS = 543


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guardian of Truth pre-flight check")
    parser.add_argument("--input", default=None, help="optional parquet/jsonl to smoke-test")
    parser.add_argument("--limit", type=int, default=8, help="rows for the smoke test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--backend", default=None, choices=["auto", "vllm", "transformers", "heuristic"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _import_report() -> dict:
    report: dict = {}
    try:
        import torch  # type: ignore

        report["torch"] = getattr(torch, "__version__", "?")
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = props.name
            report["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    except Exception as exc:  # pragma: no cover - depends on the machine
        report["torch"] = f"missing ({exc})"
        report["cuda_available"] = False
    for name in ("vllm", "transformers"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "?")
        except Exception as exc:  # pragma: no cover
            report[name] = f"missing ({exc})"
    return report


def _model_report(config: RuntimeConfig) -> dict:
    path = config.model.get("path") or ""
    model_id = config.model.get("id") or ""
    report: dict = {"model_id": model_id, "model_path": path}
    if path:
        p = Path(path)
        report["path_exists"] = p.exists()
        report["has_config"] = (p / "config.json").exists() if p.exists() else False
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["id"] = args.model
    if args.model_path:
        overrides.setdefault("model", {})["path"] = args.model_path
    if args.backend:
        overrides["backend"] = args.backend
    for item in args.set:
        key, _, raw = item.partition("=")
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = raw

    config = RuntimeConfig.load(args.config, overrides)
    report: dict = {"imports": _import_report(), "model": _model_report(config)}

    if not args.input:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    rows = read_table(args.input)[: max(1, args.limit)]
    report["rows"] = len(rows)
    started = time.monotonic()
    detector = Detector(config, backend_name=args.backend or config.raw.get("backend", "auto"))
    predictions = detector.predict_rows(rows)
    elapsed = time.monotonic() - started
    per_row = elapsed / max(1, len(rows))
    report["stats"] = detector.stats
    report["positive_rate"] = round(
        sum(p.label for p in predictions) / max(1, len(predictions)), 3
    )
    report["elapsed_sec"] = round(elapsed, 2)
    report["projected_sec_private"] = round(per_row * PRIVATE_ROWS, 1)

    warnings: list[str] = []
    if report["stats"].get("backend") == "heuristic" and args.backend != "heuristic":
        warnings.append(
            "backend fell back to rules: no model was loaded, check vLLM/CUDA/model path"
        )
    if report["projected_sec_private"] > 1500:
        warnings.append(
            "projected runtime exceeds 25 minutes; use a smaller model, fewer evidence "
            "chars, or disable the second-opinion pass"
        )
    if report["positive_rate"] in (0.0, 1.0) and len(rows) >= 8:
        warnings.append("degenerate positive rate on the smoke sample")
    report["warnings"] = warnings

    print(json.dumps(report, ensure_ascii=False, indent=2))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0 if predictions else 1


if __name__ == "__main__":
    raise SystemExit(main())
