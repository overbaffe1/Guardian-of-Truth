"""Parser for the competition transcript format.

The ``prompt`` column is a flattened dialogue where roles are marked with
``⟦SYSTEM⟧`` / ``⟦USER⟧`` / ``⟦ASSISTANT · ход N⟧`` and tool interactions are
rendered as::

    → TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}
    ← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", ...}

The parser is deliberately tolerant: real data contains pretty-printed JSON,
markdown code fences, missing colons and occasional plain-text tool outputs.
"""

from __future__ import annotations

import re
from typing import Optional

from .text_utils import lenient_json_loads, looks_like_json_start, needs_more_json
from .types import Event, ResponseTurn, ToolCall, Transcript

# ``⟦ ... ⟧`` markers.  Role text is short, anything longer is not a marker.
MARKER_RE = re.compile(r"⟦\s*([^⟧]{0,80}?)\s*⟧")

_ARROW_RIGHT = r"(?:→|->|=>|➜|➡|⇨|⇥)"
_ARROW_LEFT = r"(?:←|<-|<=|⟵|⬅|⇦)"
_NAME = r"[A-Za-z0-9_][\w.\-]*"

TOOL_CALL_RE = re.compile(
    rf"^\s*{_ARROW_RIGHT}\s*TOOL[_ ]?CALL\s*:?\s*`?({_NAME})`?\s*:?\s*(.*)$",
    re.IGNORECASE,
)
# Arrow-less fallback (some renderers drop the arrow).
TOOL_CALL_BARE_RE = re.compile(
    rf"^\s*TOOL[_ ]?CALL\s*:?\s*`?({_NAME})`?\s*:?\s*(.*)$", re.IGNORECASE
)
# The official renderer marks environment failures as ``name [ERROR]:``.
TOOL_RESP_RE = re.compile(
    rf"^\s*{_ARROW_LEFT}\s*TOOL[_ ]?RESPO?N?S?E?\s*:?\s*`?({_NAME})`?\s*"
    r"(\[\s*ERROR\s*\])?\s*:?\s*(.*)$",
    re.IGNORECASE,
)
TOOL_RESP_BARE_RE = re.compile(
    rf"^\s*TOOL[_ ]?RESPO?N?S?E?\s*:?\s*`?({_NAME})`?\s*"
    r"(\[\s*ERROR\s*\])?\s*:?\s*(.*)$",
    re.IGNORECASE,
)

_TURN_RE = re.compile(r"(?:ход|turn|step)\s*(\d+)", re.IGNORECASE)

# Fallback for renderers that use plain ``Role:`` lines instead of ⟦…⟧ markers.
PLAIN_ROLE_RE = re.compile(
    r"^[ \t]*[\[(]?[ \t]*"
    r"(SYSTEM|USER|HUMAN|CUSTOMER|CLIENT|ASSISTANT|AGENT|BOT|TOOL[ _]?(?:CALL|RESPONSE|RESULT)?)"
    r"[ \t]*[\])]?[ \t]*[:：][ \t]?",
    re.IGNORECASE | re.MULTILINE,
)

DOMAIN_HINTS = {
    "banking_knowledge": (
        "rho-bank", "rho bank", "bank account", "credit card", "debit card",
        "open_bank_account", "banking",
    ),
    "airline": (
        "airline", "flight", "reservation", "cabin", "baggage", "airport",
        "booking",
    ),
    "retail": (
        "retail", "order #", "exchange", "product", "deliver", "return",
        "item", "variant",
    ),
    "telecom": (
        "telecom", "mobile data", "roaming", "sim card", "line", "mms",
        "network", "plan", "bill",
    ),
}


def _classify_marker(text: str) -> tuple[str, Optional[int]]:
    upper = text.strip().upper()
    turn: Optional[int] = None
    m = _TURN_RE.search(text)
    if m:
        turn = int(m.group(1))
    if upper.startswith("SYSTEM"):
        return "system", turn
    if upper.startswith(("USER", "HUMAN", "CUSTOMER", "CLIENT")):
        return "user", turn
    if upper.startswith(("ASSISTANT", "AGENT", "BOT")):
        return "assistant", turn
    if upper.startswith("TOOL"):
        return "other", turn
    return "other", turn


def _consume_json_block(lines: list[str], start: int, first: str) -> tuple[str, int]:
    """Accumulate lines until the JSON payload started in ``first`` balances."""
    text = first
    if not looks_like_json_start(first) or not needs_more_json(first):
        return text, start + 1
    j = start + 1
    while j < len(lines) and j < start + 400:
        text += "\n" + lines[j]
        j += 1
        if not needs_more_json(text):
            break
    return text, j


def _consume_indented_block(lines: list[str], start: int, first: str, indent: int) -> tuple[str, int]:
    """Collect a plain-text multi-line tool output rendered with indentation."""
    text = first
    j = start + 1
    collected = 0
    while j < len(lines) and collected < 400 and len(text) < 60_000:
        nxt = lines[j]
        if not nxt.strip():
            break
        if TOOL_CALL_RE.match(nxt) or TOOL_RESP_RE.match(nxt) or TOOL_CALL_BARE_RE.match(nxt):
            break
        if MARKER_RE.search(nxt):
            break
        leading = len(nxt) - len(nxt.lstrip())
        if leading <= indent:
            break
        text += "\n" + nxt.strip()
        collected += 1
        j += 1
    return text, j


def _match_tool_line(line: str) -> Optional[tuple[str, str, str, bool]]:
    """Return ``(kind, name, rest, error)`` when the line is a tool call/response."""
    for regex, kind in (
        (TOOL_CALL_RE, "tool_call"),
        (TOOL_CALL_BARE_RE, "tool_call"),
        (TOOL_RESP_RE, "tool_response"),
        (TOOL_RESP_BARE_RE, "tool_response"),
    ):
        m = regex.match(line)
        if m:
            name = m.group(1)
            rest = (m.group(3) if kind == "tool_response" else m.group(2)) or ""
            error = kind == "tool_response" and bool(m.group(2))
            if error is False and kind == "tool_response" and rest.lstrip().lower().startswith("[error]"):
                error = True
                rest = rest.lstrip()[len("[error]") :].lstrip(" :")
            # ``TOOL_CALL: {...}`` without a name is not a valid tool line.
            if not name or name.upper() in {"CALL", "RESPONSE", "TOOL"}:
                continue
            return kind, name, rest, error
    return None


def _append_assistant_segment(
    events: list[Event],
    content: str,
    turn: Optional[int],
    order_start: int,
    caller: str = "assistant",
) -> int:
    lines = content.split("\n")
    buffer: list[str] = []
    order = order_start

    def flush_text() -> None:
        nonlocal buffer, order
        text = "\n".join(buffer).strip()
        if text:
            kind = "user" if caller == "user" else "assistant"
            events.append(Event(kind=kind, content=text, turn=turn, order=order))
            order += 1
        buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]
        matched = _match_tool_line(line)
        if matched is None:
            buffer.append(line)
            i += 1
            continue

        flush_text()
        kind, name, rest, error = matched
        indent = len(line) - len(line.lstrip())
        if kind == "tool_call":
            args_text, nxt = _consume_json_block(lines, i, rest)
            args, ok = lenient_json_loads(args_text)
            events.append(
                Event(
                    kind="tool_call",
                    turn=turn,
                    order=order,
                    tool_call=ToolCall(
                        name=name,
                        raw_args=args_text.strip(),
                        args=args if isinstance(args, dict) else None,
                        parse_ok=bool(ok and isinstance(args, dict)),
                        raw_line=line.strip(),
                        caller=caller,
                    ),
                )
            )
            order += 1
        else:
            if looks_like_json_start(rest):
                payload, nxt = _consume_json_block(lines, i, rest)
            else:
                payload, nxt = _consume_indented_block(lines, i, rest, indent)
            events.append(
                Event(
                    kind="tool_response",
                    content=payload.strip(),
                    tool_name=name,
                    turn=turn,
                    order=order,
                    error=error,
                )
            )
            order += 1
        i = max(nxt, i + 1)

    flush_text()
    return order


def parse_transcript(text: str) -> Transcript:
    """Parse the ``prompt`` column into events."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    events: list[Event] = []
    order = 0

    matches = list(MARKER_RE.finditer(raw))
    if not matches:
        plain = list(PLAIN_ROLE_RE.finditer(raw))
        if len(plain) >= 2:
            # ``System: … User: … Assistant: …`` layout.
            matches = plain
            preamble = raw[: matches[0].start()].strip()
            if len(preamble) > 40:
                events.append(Event(kind="system", content=preamble, order=order))
                order += 1
            for idx, m in enumerate(matches):
                role = _classify_marker(m.group(1))[0]
                content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
                content = raw[m.end() : content_end]
                if role in ("assistant", "other"):
                    order = _append_assistant_segment(events, content, None, order)
                elif role == "user":
                    order = _append_assistant_segment(events, content, None, order, caller="user")
                elif content.strip():
                    events.append(Event(kind=role, content=content.strip(), order=order))
                    order += 1
            transcript = Transcript(events=events, raw=raw, system_text="")
            transcript.system_text = "\n\n".join(
                e.content for e in transcript.events if e.kind == "system"
            )
            transcript.domain = detect_domain(transcript)
            return transcript
        # No markers at all: keep the whole prompt as system context.
        events.append(Event(kind="system", content=raw.strip(), order=0))
    else:
        # Text before the first marker is usually a ``prompt:`` label or a
        # preamble; keep it as system context if it is meaningful.
        preamble = raw[: matches[0].start()].strip()
        if len(preamble) > 40:
            events.append(Event(kind="system", content=preamble, order=order))
            order += 1

        for idx, m in enumerate(matches):
            role, turn = _classify_marker(m.group(1))
            content_start = m.end()
            content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
            content = raw[content_start:content_end]
            if role in ("assistant", "other"):
                order = _append_assistant_segment(events, content, turn, order)
            elif role == "user":
                # Customers may call tools too (telecom); keep their calls and
                # results as first-class events instead of burying them in text.
                order = _append_assistant_segment(events, content, turn, order, caller="user")
            else:
                content = content.strip()
                if content:
                    events.append(
                        Event(kind=role, content=content, turn=turn, order=order)
                    )
                    order += 1

    transcript = Transcript(events=events, raw=raw, system_text="")
    transcript.system_text = "\n\n".join(
        e.content for e in transcript.events if e.kind == "system"
    )
    transcript.domain = detect_domain(transcript)
    return transcript


def parse_response(text: str) -> ResponseTurn:
    """Parse the ``response`` column (a single agent turn)."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    events: list[Event] = []
    turn: Optional[int] = None
    matches = list(MARKER_RE.finditer(raw))
    if matches:
        # Keep only assistant segments (usually exactly one).
        for idx, m in enumerate(matches):
            role, marker_turn = _classify_marker(m.group(1))
            if marker_turn is not None:
                turn = marker_turn
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
            chunk = raw[m.end() : end]
            if role in ("assistant", "other"):
                _append_assistant_segment(events, chunk, marker_turn, len(events))
            elif chunk.strip():
                events.append(Event(kind="assistant", content=chunk.strip()))
    else:
        _append_assistant_segment(events, raw, None, 0)

    texts = [e.content.strip() for e in events if e.kind == "assistant" and e.content.strip()]
    calls = [e.tool_call for e in events if e.kind == "tool_call" and e.tool_call]
    return ResponseTurn(raw=raw, text="\n\n".join(texts).strip(), tool_calls=calls, turn=turn)


def detect_domain(transcript: Transcript) -> str:
    system = (transcript.system_text or "").lower()
    tool_names = " ".join(
        e.tool_call.name.lower() for e in transcript.tool_calls if e.tool_call
    )
    scores: dict[str, int] = {}
    for domain, hints in DOMAIN_HINTS.items():
        score = 0
        for hint in hints:
            if hint in system:
                score += 2
            if hint in tool_names:
                score += 3
        scores[domain] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def extract_policy(system_text: str) -> str:
    """Return the ``<policy>`` section when present, otherwise the system text.

    The instructions often *mention* the tag in prose ("according to the
    <policy> provided below"), so the opening tag must start a line; otherwise
    the match would begin inside the instructions and swallow them.
    """
    if not system_text:
        return ""
    m = re.search(
        r"(?:^|\n)[ \t]*<policy>[ \t]*\n(.*?)\n[ \t]*</policy>",
        system_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        m = re.search(r"<policy>(.*?)</policy>", system_text, re.IGNORECASE | re.DOTALL)
    if m and len(m.group(1).strip()) >= 10:
        return m.group(1).strip()
    return system_text.strip()


def extract_instructions(system_text: str) -> str:
    m = re.search(r"<instructions>(.*?)</instructions>", system_text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def iter_turns(transcript: Transcript) -> list[list[Event]]:
    """Group events into assistant turns (each turn keeps its tool interactions)."""
    groups: list[list[Event]] = []
    current: list[Event] = []
    for e in transcript.events:
        if e.kind == "system":
            continue
        if e.kind == "assistant":
            if current:
                groups.append(current)
            current = [e]
        else:
            current.append(e)
    if current:
        groups.append(current)
    return groups
