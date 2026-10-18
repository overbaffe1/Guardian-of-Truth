#!/usr/bin/env python3
"""Inspect one example: parsed events, signals, packed context and judge prompt.

    python scripts/inspect.py --input test.parquet --index 0 --show-prompt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.config import RuntimeConfig  # noqa: E402
from guardian.io_utils import read_table  # noqa: E402
from guardian.pipeline import Detector  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args(argv)

    rows = read_table(args.input)
    row = None
    if args.id is not None:
        row = next((r for r in rows if str(r["id"]) == str(args.id)), None)
    else:
        row = rows[args.index]
    if row is None:
        print("Row not found", file=sys.stderr)
        return 2

    detector = Detector(RuntimeConfig.load(args.config), backend_name="heuristic")
    explanation = detector.explain(str(row["prompt"]), str(row["response"]))

    print(f"id: {row['id']}")
    print(f"domain: {explanation['domain']}")
    print(f"prompt chars: {len(str(row['prompt']))}, response chars: {len(str(row['response']))}")
    print(f"packed evidence chars: {explanation['evidence_chars']}")

    item = detector.prepare("debug", str(row["prompt"]), str(row["response"]))
    print(f"events: {[(e.kind, (e.tool_call.name if e.tool_call else e.tool_name) or '') for e in item.transcript.events][:40]}")
    print(f"tool calls in prompt: {len(item.transcript.tool_calls)}, tool responses: {len(item.transcript.tool_responses)}")
    print(f"response tool calls: {[(c.name, c.args, c.parse_ok) for c in item.response.tool_calls]}")
    print("\nsignals:")
    for signal in explanation["signals"]:
        print(f"  [{signal['severity']}] {signal['kind']}: {signal['message']}")
    if not explanation["signals"]:
        print("  none")

    if args.show_evidence and item.messages:
        print("\n--- CONTEXT ---")
        print(item.messages[-1]["content"].split("[RESPONSE UNDER AUDIT]")[0])
    if args.show_prompt and item.messages:
        print("\n--- JUDGE MESSAGES ---")
        print(json.dumps(item.messages, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
