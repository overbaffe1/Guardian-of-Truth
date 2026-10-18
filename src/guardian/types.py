"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Kind = Literal["system", "user", "assistant", "tool_call", "tool_response", "other"]
Severity = Literal["hard", "soft"]


@dataclass
class ToolCall:
    """A single ``→ TOOL_CALL <name>: {json}`` entry."""

    name: str
    raw_args: str = ""
    args: Optional[dict[str, Any]] = None
    parse_ok: bool = False
    #: tool names may be decorated by the transcript renderer, e.g. ``name``
    #: surrounded by backticks.  The clean name is stored in ``name``.
    raw_line: str = ""

    def args_preview(self, limit: int = 600) -> str:
        if self.parse_ok and self.args is not None:
            import json

            text = json.dumps(self.args, ensure_ascii=False, sort_keys=True)
        else:
            text = self.raw_args or ""
        return text[:limit]


@dataclass
class Event:
    """One parsed element of a transcript."""

    kind: Kind
    content: str = ""
    turn: Optional[int] = None
    tool_call: Optional[ToolCall] = None
    tool_name: Optional[str] = None  # for tool_response
    order: int = 0

    def short(self, limit: int = 400) -> str:
        text = self.content if self.kind != "tool_call" else (
            f"{self.tool_call.name}: {self.tool_call.args_preview(limit)}"
            if self.tool_call
            else ""
        )
        text = " ".join(text.split())
        return text[:limit]


@dataclass
class Transcript:
    """Parsed ``prompt`` column."""

    events: list[Event] = field(default_factory=list)
    raw: str = ""
    system_text: str = ""
    domain: str = "unknown"

    # ---------------------------------------------------------------- helpers
    def of_kind(self, *kinds: str) -> list[Event]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    @property
    def users(self) -> list[Event]:
        return [e for e in self.events if e.kind == "user"]

    @property
    def assistant_texts(self) -> list[Event]:
        return [e for e in self.events if e.kind == "assistant" and e.content.strip()]

    @property
    def tool_calls(self) -> list[Event]:
        return [e for e in self.events if e.kind == "tool_call"]

    @property
    def tool_responses(self) -> list[Event]:
        return [e for e in self.events if e.kind == "tool_response"]

    @property
    def last_user_index(self) -> int:
        idx = -1
        for i, e in enumerate(self.events):
            if e.kind == "user":
                idx = i
        return idx

    def last_user_text(self) -> str:
        users = self.users
        return users[-1].content if users else ""

    def full_text(self) -> str:
        return self.raw


@dataclass
class ResponseTurn:
    """Parsed ``response`` column: zero or more tool calls plus optional text."""

    raw: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    turn: Optional[int] = None

    @property
    def is_tool_call_only(self) -> bool:
        return bool(self.tool_calls) and not self.text.strip()

    def summary(self, limit: int = 2000) -> str:
        parts: list[str] = []
        if self.text.strip():
            parts.append(self.text.strip())
        for tc in self.tool_calls:
            parts.append(f"TOOL_CALL {tc.name}: {tc.args_preview(limit)}")
        return "\n".join(parts)[:limit]


@dataclass
class Signal:
    """A deterministic pre-check result.

    ``hard`` signals are sufficient to output label 1 on their own (the task
    statement lists all of them explicitly as hallucinations).  ``soft``
    signals are hints passed to the LLM judge; they never decide alone.
    """

    kind: str
    severity: Severity
    message: str
    evidence: str = ""

    def as_hint(self) -> str:
        prefix = "STRONG" if self.severity == "hard" else "weak"
        ev = f" | evidence: {self.evidence}" if self.evidence else ""
        return f"- [{prefix}] {self.kind}: {self.message}{ev}"


@dataclass
class JudgeResult:
    label: int
    confidence: float = 0.5
    hallucination_type: str = "none"
    reason: str = ""
    raw: str = ""
    votes: list[int] = field(default_factory=list)
    failed: bool = False


@dataclass
class Prediction:
    id: str
    label: int
    score: float = 0.0
    judge_label: Optional[int] = None
    judge_confidence: Optional[float] = None
    signals: list[Signal] = field(default_factory=list)
    reason: str = ""
    backend: str = ""

    def to_row(self, with_details: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {"id": self.id, "label": int(self.label)}
        if with_details:
            row.update(
                {
                    "score": round(float(self.score), 6),
                    "judge_label": self.judge_label,
                    "judge_confidence": self.judge_confidence,
                    "signals": "; ".join(s.kind for s in self.signals),
                    "reason": (self.reason or "")[:2000],
                    "backend": self.backend,
                }
            )
        return row
