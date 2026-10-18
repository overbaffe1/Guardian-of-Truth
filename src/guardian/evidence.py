"""Budgeted evidence packing.

Prompts in this task are long (median ~32k characters, max >230k), while the
judge model has a finite context window.  Blind truncation is dangerous: the
ground truth is usually buried in one of the tool responses.  We therefore
rank tool outputs by lexical overlap with the audited response (identifiers
and numbers weigh more), keep the most recent ones, and render everything in
chronological order so the judge can follow the dialogue.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .text_utils import (
    compact_json,
    content_words,
    estimate_tokens,
    overlap_score,
    truncate_middle,
)
from .toolspec import ToolSpec, catalogue_text, extract_tools_block
from .transcript import extract_policy
from .types import Event, ResponseTurn, Transcript


@dataclass
class EvidenceConfig:
    max_chars: int = 30_000          # ~10k tokens
    policy_ratio: float = 0.40
    tools_ratio: float = 0.08
    max_tool_response_chars: int = 3_500
    max_tool_call_chars: int = 700
    max_assistant_chars: int = 700
    max_user_chars: int = 1_800
    keep_last_tool_responses: int = 4
    max_selected_tool_responses: int = 40
    max_json_items: int = 6

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "EvidenceConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Evidence:
    text: str
    policy_excerpt: str
    tools_excerpt: str
    selected_tool_responses: int = 0
    total_tool_responses: int = 0
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def approx_tokens(self) -> int:
        return estimate_tokens(self.text)


def _query_text(response: ResponseTurn) -> str:
    parts = [response.text]
    for tc in response.tool_calls:
        parts.append(tc.name)
        if tc.parse_ok and tc.args is not None:
            parts.append(json.dumps(tc.args, ensure_ascii=False))
        else:
            parts.append(tc.raw_args)
    return "\n".join(p for p in parts if p)


def _shrink_policy(policy: str, budget: int, query: str) -> str:
    if len(policy) <= budget:
        return policy
    # Keep the header and the sections that share vocabulary with the response.
    sections = re.split(r"\n(?=#{1,4}\s)", policy)
    head_budget = min(4_000, budget // 3)
    chosen: list[str] = [truncate_middle(sections[0], head_budget)]
    used = len(chosen[0])
    scored = []
    for idx, section in enumerate(sections[1:], start=1):
        scored.append((overlap_score(query, section), idx, section))
    scored.sort(key=lambda x: (-x[0], x[1]))
    for score, _, section in scored:
        if used + len(section) > budget:
            continue
        chosen.append(section)
        used += len(section)
    if len(chosen) == 1:
        return truncate_middle(policy, budget)
    return "\n".join(chosen)[:budget]


def _matching_items(items: list[Any], query_words: set[str], limit: int) -> list[Any]:
    picked: list[Any] = []
    for item in items:
        text = json.dumps(item, ensure_ascii=False).lower() if not isinstance(item, str) else item.lower()
        if any(w in text for w in query_words):
            picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def _compress_json(value: Any, query: str, budget: int, max_items: int) -> str:
    query_words = set(content_words(query))
    query_nums = set(re.findall(r"\d[\d,]*", query))

    def prune(node: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "…"
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, val in node.items():
                key_low = str(key).lower()
                relevant = (
                    key_low in query.lower()
                    or (isinstance(val, str) and any(w in val.lower() for w in query_words))
                    or (isinstance(val, (int, float)) and str(val) in query_nums)
                )
                if isinstance(val, (dict, list)) and not relevant:
                    pruned = prune(val, depth + 1)
                    if pruned in ({}, []):
                        continue
                    # keep short nested structures, drop long irrelevant ones
                    if len(json.dumps(pruned, ensure_ascii=False)) > max(400, budget // 8):
                        continue
                    out[str(key)] = pruned
                else:
                    out[str(key)] = val
            return out
        if isinstance(node, list):
            if len(node) <= max_items:
                return [prune(v, depth + 1) for v in node]
            keep = node[:3]
            keep += _matching_items(node[3:], query_words, max_items - 3)
            return [prune(v, depth + 1) for v in keep]
        return node

    compact = prune(value)
    text = compact_json(compact, limit=budget)
    if len(text) > budget:
        text = truncate_middle(text, budget)
    return text


def compress_tool_response(
    content: str, tool_name: str, query: str, budget: int, max_items: int = 6
) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    if len(text) <= budget:
        return text
    stripped = text.strip("`").strip()
    if stripped[:1] in "{[":
        try:
            value = json.loads(stripped)
            return _compress_json(value, query, budget, max_items)
        except Exception:
            pass
    return truncate_middle(text, budget)


def select_tool_responses(
    transcript: Transcript,
    response: ResponseTurn,
    cfg: EvidenceConfig,
    budget: int,
) -> tuple[list[int], dict[int, str]]:
    """Return (selected event orders, compressed payloads by order)."""
    query = _query_text(response)
    tool_events = transcript.tool_responses
    if not tool_events:
        return [], {}

    scored: list[tuple[float, int, Event]] = []
    n = len(tool_events)
    # Unique identifier-like tokens from the response (bounded for speed).
    query_tokens = list(dict.fromkeys(
        t.lower() for t in re.findall(r"[A-Za-z0-9_\-]{5,}", query)
    ))[:200]
    for i, event in enumerate(tool_events):
        score = 2.5 * overlap_score(query, event.content)
        if i >= n - cfg.keep_last_tool_responses:
            score += 1.5
        # Identifiers from the response inside this payload are very strong.
        if query_tokens:
            content_low = event.content.lower()
            hits = sum(1 for token in query_tokens if token in content_low)
            score += 0.4 * min(hits, 5)
        scored.append((score, i, event))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    selected: list[int] = []
    payloads: dict[int, str] = {}
    used = 0
    for score, i, event in scored:
        if len(selected) >= cfg.max_selected_tool_responses:
            break
        remaining = budget - used
        if remaining < 300 and selected:
            break
        per_item = max(300, min(cfg.max_tool_response_chars, remaining))
        compressed = compress_tool_response(
            event.content, event.tool_name or "", query, per_item, cfg.max_json_items
        )
        if not compressed:
            continue
        selected.append(event.order)
        payloads[event.order] = compressed
        used += len(compressed)
        if used >= budget:
            break

    # Always keep the most recent tool output even if it scored low.
    if tool_events:
        last = tool_events[-1]
        if last.order not in payloads:
            compressed = compress_tool_response(
                last.content, last.tool_name or "", query, min(cfg.max_tool_response_chars, budget)
            )
            if compressed:
                payloads[last.order] = compressed
                selected.append(last.order)
    return sorted(selected), payloads


def build_evidence(
    transcript: Transcript,
    response: ResponseTurn,
    specs: dict[str, ToolSpec],
    config: Optional[EvidenceConfig] = None,
) -> Evidence:
    cfg = config or EvidenceConfig()
    total_budget = cfg.max_chars
    query = _query_text(response)

    # -- 1. policy -----------------------------------------------------------
    policy_budget = int(total_budget * cfg.policy_ratio)
    policy_full = extract_policy(transcript.system_text)
    policy = _shrink_policy(policy_full, policy_budget, query)

    # -- 2. tool catalogue ---------------------------------------------------
    tools_budget = int(total_budget * cfg.tools_ratio)
    tools_text = catalogue_text(specs, max_desc=110, max_chars=tools_budget)
    if not specs:
        # The catalogue could not be parsed: keep the raw block so the judge
        # can still check which tools exist.
        raw_block = extract_tools_block(transcript.system_text)
        if raw_block:
            tools_text = raw_block
    if len(tools_text) > tools_budget:
        tools_text = truncate_middle(tools_text, tools_budget)

    # -- 3. conversation -----------------------------------------------------
    overhead = len(policy) + len(tools_text) + 200
    # User turns are always kept (they carry the request), so reserve space for
    # them up front instead of letting them push the text over the budget.
    user_reserve = sum(
        min(len(e.content), cfg.max_user_chars) + 24
        for e in transcript.events
        if e.kind == "user"
    )
    convo_budget = max(2_000, total_budget - overhead - user_reserve)
    selected_orders, payloads = select_tool_responses(transcript, response, cfg, convo_budget)
    selected_set = set(selected_orders)

    rendered: list[str] = []
    used_convo = 0
    truncated = len(policy) < len(policy_full)

    for event in transcript.events:
        if event.kind == "system":
            continue
        if event.kind == "user":
            body = truncate_middle(event.content, cfg.max_user_chars)
            chunk = f"⟦USER⟧\n{body}"
        elif event.kind == "assistant":
            body = truncate_middle(event.content, cfg.max_assistant_chars)
            turn_label = f" · ход {event.turn}" if event.turn is not None else ""
            chunk = f"⟦ASSISTANT{turn_label}⟧\n{body}"
        elif event.kind == "tool_call" and event.tool_call:
            tc = event.tool_call
            args = tc.args_preview(cfg.max_tool_call_chars) if tc.parse_ok else truncate_middle(
                tc.raw_args, cfg.max_tool_call_chars
            )
            chunk = f"  → TOOL_CALL {tc.name}: {args}"
        elif event.kind == "tool_response":
            if event.order not in selected_set:
                chunk = f"  ← TOOL_RESPONSE {event.tool_name or ''}: [omitted to fit context]"
            else:
                body = payloads.get(event.order, "")
                chunk = f"  ← TOOL_RESPONSE {event.tool_name or ''}: {body}"
        else:
            continue

        if used_convo + len(chunk) > convo_budget:
            truncated = True
            if event.kind == "tool_response" and event.order in selected_set:
                # shrink this payload instead of dropping it
                remaining = max(200, convo_budget - used_convo - 80)
                body = truncate_middle(payloads.get(event.order, ""), remaining)
                chunk = f"  ← TOOL_RESPONSE {event.tool_name or ''}: {body}"
            elif event.kind == "user":
                # a user turn is never dropped, only shrunk
                remaining = max(120, convo_budget - used_convo - 16)
                chunk = f"⟦USER⟧\n{truncate_middle(event.content, remaining)}"
            else:
                continue
        rendered.append(chunk)
        used_convo += len(chunk)

    header = (
        "⟦SYSTEM⟧\n"
        f"{policy}\n\n"
        "[AVAILABLE TOOLS]\n"
        f"{tools_text}\n\n"
    )
    text = header + "\n\n".join(rendered)
    # Final hard clamp: drop the oldest conversation chunks until the budget is
    # respected, so a burst of long user turns can never overflow the context.
    while len(text) > total_budget and rendered:
        truncated = True
        rendered.pop(0)
        text = header + "\n\n".join(rendered)
    return Evidence(
        text=text,
        policy_excerpt=policy,
        tools_excerpt=tools_text,
        selected_tool_responses=len(selected_orders),
        total_tool_responses=len(transcript.tool_responses),
        truncated=truncated,
        meta={
            "chars": len(text),
            "approx_tokens": estimate_tokens(text),
            "domain": transcript.domain,
        },
    )


def response_as_text(response: ResponseTurn, limit: int = 8_000) -> str:
    parts: list[str] = []
    if response.text.strip():
        parts.append(response.text.strip())
    for tc in response.tool_calls:
        args = tc.args_preview(limit) if tc.parse_ok else truncate_middle(tc.raw_args, limit)
        parts.append(f"→ TOOL_CALL {tc.name}: {args}")
    return "\n".join(parts)[:limit]
