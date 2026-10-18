#!/usr/bin/env python3
"""Build a synthetic development set in the competition format from tau2-bench.

tau2-bench (MIT, https://github.com/sierra-research/tau2-bench) is the open
benchmark whose domains, policies and tool inventories are used by this task.
The script replays real successful trajectories, renders them with the
competition markers, and optionally applies controlled corruptions to obtain
unambiguous positive examples.

The result is *not* a replacement for the official public split: labels are
synthetic by construction and the text distribution is only an approximation.
It is meant for smoke-testing the parser, the rule engine and the end-to-end
pipeline without the private data.

Example:
    python scripts/make_dev_from_tau2.py \
        --tau2-root /tmp/tau2 --out data/dev_synthetic.jsonl --per-domain 40
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> "
    "provided below.\nIn each turn you can either:\n- Send a message to the user.\n"
    "- Make a tool call.\nYou cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)

DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")


def extract_tool_names(tau2_root: Path, domain: str) -> list[str]:
    path = tau2_root / "src" / "tau2" / "domains" / domain / "tools.py"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    names: list[str] = []
    for match in re.finditer(r"@is_tool\([^)]*\)\s*\n\s*(?:async\s+)?def\s+(\w+)\(", text):
        names.append(match.group(1))
    # de-duplicate, keep order
    return list(dict.fromkeys(names))


def tool_catalogue(tau2_root: Path, domain: str, observed: Iterable[str]) -> str:
    names = list(dict.fromkeys(list(observed) + extract_tool_names(tau2_root, domain)))
    if not names:
        return "(no tools)"
    return "\n".join(f"- {name}: tool available to the agent" for name in names)


def render_system(policy: str, tools_block: str) -> str:
    return (
        "⟦SYSTEM⟧\n"
        f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
        f"<policy>\n{policy.strip()}\n</policy>\n\n"
        f"[AVAILABLE TOOLS]\n{tools_block}"
    )


def render_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tau2 messages into (event_role, text) chunks."""
    call_id_to_name: dict[str, str] = {}
    rendered: list[dict[str, Any]] = []
    turn = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "assistant":
            turn += 1
            lines: list[str] = []
            if str(content).strip():
                lines.append(str(content).strip())
            for call in message.get("tool_calls") or []:
                call_id_to_name[call.get("id", "")] = call.get("name", "")
                args = call.get("arguments")
                args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                lines.append(f"\t→ TOOL_CALL {call.get('name')}: {args_text}")
            rendered.append(
                {
                    "role": "assistant",
                    "turn": turn,
                    "text": "\n".join(lines),
                    "kind": "tool_call" if message.get("tool_calls") else "text",
                }
            )
        elif role == "user":
            rendered.append({"role": "user", "turn": None, "text": str(content).strip(), "kind": "text"})
        elif role == "tool":
            name = call_id_to_name.get(message.get("id", ""), "tool")
            rendered.append(
                {
                    "role": "tool",
                    "turn": turn,
                    "text": str(content).strip(),
                    "kind": "tool_response",
                    "name": name,
                }
            )
    return rendered


def flatten(rendered: list[dict[str, Any]], upto: int | None = None) -> str:
    parts: list[str] = []
    for chunk in rendered[:upto]:
        if chunk["role"] == "user":
            parts.append(f"⟦USER⟧\n{chunk['text']}")
        elif chunk["role"] == "assistant":
            parts.append(f"⟦ASSISTANT · ход {chunk['turn']}⟧\n{chunk['text']}")
        elif chunk["role"] == "tool":
            parts.append(f"\t← TOOL_RESPONSE {chunk['name']}: {chunk['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Corruptions (unambiguous positives)
# ---------------------------------------------------------------------------

MONEY_RE = re.compile(r"(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars?))", re.I)
CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{4,7}\b")
FABRICATED = (
    " Also, your booking includes complimentary lounge access and a free checked bag.",
    " For your convenience, we have already emailed the receipt to the address on file.",
    " Please note that a 15% loyalty discount was automatically applied to this transaction.",
)
FAKE_TOOL = (
    ("issue_full_refund", {"reservation_id": "UNKNOWN01", "amount": 999}),
    ("apply_loyalty_discount", {"order_id": "UNKNOWN01", "percent": 15}),
    ("sync_customer_profile", {"customer_id": "UNKNOWN01", "force": True}),
)


def corrupt_response(
    response: dict[str, Any],
    domain: str,
    rng: random.Random,
) -> tuple[dict[str, Any], str] | None:
    text = response["text"]
    kind = response["kind"]
    options = ["fabricate", "tool"]
    if kind == "text" and MONEY_RE.search(text):
        options.append("amount")
    if kind == "text" and CODE_RE.search(text):
        options.append("code")
    choice = rng.choice(options)

    if choice == "fabricate":
        return (
            {**response, "text": text.rstrip() + " " + rng.choice(FABRICATED)},
            "unsupported: an invented fact was appended",
        )
    if choice == "amount":
        match = MONEY_RE.search(text)
        if not match:
            return None
        original = match.group(0)
        number = re.search(r"\d[\d,]*(?:\.\d+)?", original)
        if not number:
            return None
        try:
            value = float(number.group(0).replace(",", ""))
        except ValueError:
            return None
        new_value = round(value * 1.7 + 13, 2)
        replacement = f"{new_value:.2f} USD" if "USD" in original else f"${new_value:.2f}"
        return (
            {**response, "text": text.replace(original, replacement, 1)},
            "contradiction: a monetary amount was changed",
        )
    if choice == "code":
        token = CODE_RE.search(text)
        if not token:
            return None
        fake = "QX" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))
        return (
            {**response, "text": text.replace(token.group(0), fake, 1)},
            "contradiction: an identifier was replaced with a fabricated one",
        )
    if choice == "tool":
        name, args = rng.choice(FAKE_TOOL)
        call = f"\t→ TOOL_CALL {name}: {json.dumps(args)}"
        new_text = (text + "\n" + call) if text.strip() else call
        return (
            {**response, "text": new_text, "kind": "tool_call"},
            "action error: a tool that is not in [AVAILABLE TOOLS] was called",
        )
    return None


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-domain", type=int, default=40)
    parser.add_argument("--max-turns-per-sim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-corrupted", action="store_true", default=True)
    parser.add_argument("--no-corrupted", dest="include_corrupted", action="store_false")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    results_dir = args.tau2_root / "data" / "tau2" / "results" / "final"
    files = sorted(glob.glob(str(results_dir / "*_4trials.json")))
    if not files:
        print(f"No result files under {results_dir}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    counter = 0
    for domain in DOMAINS:
        domain_files = [f for f in files if f"_{domain}_" in Path(f).name or f"_{domain}-" in Path(f).name]
        # airline/retail/telecom have result dumps; banking_knowledge may not.
        if not domain_files:
            continue
        rng.shuffle(domain_files)
        per_domain = 0
        for file_path in domain_files:
            if per_domain >= args.per_domain:
                break
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            policy = (data.get("info", {}).get("environment_info", {}) or {}).get("policy", "")
            if not policy:
                continue
            simulations = data.get("simulations", [])
            rng.shuffle(simulations)
            for sim in simulations:
                if per_domain >= args.per_domain:
                    break
                reward = (sim.get("reward_info") or {}).get("reward")
                if reward != 1.0:
                    continue
                rendered = render_messages(sim.get("messages", []))
                observed = [
                    c["name"] for c in rendered if c["role"] == "tool" and c.get("name")
                ]
                tools_block = tool_catalogue(args.tau2_root, domain, observed)
                system = render_system(policy, tools_block)
                assistant_idx = [i for i, c in enumerate(rendered) if c["role"] == "assistant"]
                if not assistant_idx:
                    continue
                chosen = assistant_idx[-args.max_turns_per_sim :]
                for idx in chosen:
                    response = rendered[idx]
                    if not response["text"].strip():
                        continue
                    prompt = system + "\n\n" + flatten(rendered, upto=idx)
                    counter += 1
                    records.append(
                        {
                            "id": f"syn_{domain}_{counter:05d}_0",
                            "prompt": prompt,
                            "response": f"⟦ASSISTANT · ход {response['turn']}⟧\n{response['text']}",
                            "label": 0,
                            "explanation": "replayed successful tau2 trajectory (synthetic negative)",
                            "meta": {"domain": domain, "source": Path(file_path).name,
                                     "task_id": sim.get("task_id"), "trial": sim.get("trial")},
                        }
                    )
                    if args.include_corrupted:
                        corrupted = corrupt_response(response, domain, rng)
                        if corrupted is not None:
                            new_response, why = corrupted
                            counter += 1
                            records.append(
                                {
                                    "id": f"syn_{domain}_{counter:05d}_1",
                                    "prompt": prompt,
                                    "response": f"⟦ASSISTANT · ход {response['turn']}⟧\n{new_response['text']}",
                                    "label": 1,
                                    "explanation": why,
                                    "meta": {"domain": domain, "source": Path(file_path).name,
                                             "task_id": sim.get("task_id"), "trial": sim.get("trial"),
                                             "mutation": why.split(":")[0]},
                                }
                            )
                    per_domain += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    positives = sum(r["label"] for r in records)
    print(
        json.dumps(
            {
                "written": str(args.out),
                "rows": len(records),
                "positives": positives,
                "negatives": len(records) - positives,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
