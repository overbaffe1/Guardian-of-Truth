"""Deterministic pre-checks.

The task statement explicitly lists several hallucination categories that do
not require semantic judgement:

* calling a tool that is not in ``[AVAILABLE TOOLS]``;
* invalid tool arguments (missing required fields, wrong types/enums);
* referencing an entity / id that does not exist in the provided data;
* violating a mandatory policy rule (e.g. mutating state without the required
  confirmation);
* false refusal of a request the tools and policy allow.

Checks that can be decided with certainty are emitted as ``hard`` signals and
short-circuit the LLM.  Everything that needs world knowledge or careful
reading is emitted as a ``soft`` hint that is passed to the judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .text_utils import canonical_number, norm_token, numbers_normalized
from .toolspec import ToolSpec, validate_tool_call
from .types import ResponseTurn, Signal, Transcript

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SignalsConfig:
    enable_tool_availability: bool = True
    enable_schema_checks: bool = True
    enable_identifier_grounding: bool = True
    identifier_severity: str = "hard"  # "hard" | "soft"
    enable_numeric_hints: bool = True
    enable_confirmation_check: bool = True
    enable_refusal_check: bool = True
    enable_duplicate_check: bool = True
    enable_action_claim_check: bool = True
    enable_request_alignment: bool = True
    enable_success_after_failure: bool = True
    max_soft_hints: int = 12

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SignalsConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

ID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    (
        "phone",
        re.compile(
            r"(?<![\d.])(?:\+\d{1,3}[\s.\-]?)?"
            r"(?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])"
            r"\d{3,4}[\s.\-]\d{2,4}(?![\d.])"
        ),
    ),
    ("doc_id", re.compile(r"\bdoc_[A-Za-z0-9_()\-]{3,}\b")),
    ("order_id", re.compile(r"#\s?[A-Za-z0-9]{5,}\b")),
    (
        "code",
        re.compile(r"\b(?=[A-Z0-9][A-Z0-9_-]{4,19}\b)(?=[A-Z0-9_-]*\d)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9_-]{5,20}\b"),
    ),
]
_ID_ARG_RE = re.compile(r"(?:^|_)(?:id|ids|number|no|code|ref|reference)$", re.IGNORECASE)
#: arguments that identify the *object* of an action (not payment methods,
#: addresses or other attributes the user never has to mention)
_TARGET_ID_KEYS = {
    "reservation_id", "order_id", "line_id", "account_id", "customer_id",
    "user_id", "item_id", "booking_id", "ticket_id", "subscription_id",
    "transaction_id", "loan_id", "claim_id", "case_id", "task_id", "flight_id",
}
_NON_TARGET_ID_KEYS = {
    "payment_id", "payment_method_id", "card_id", "address_id", "billing_id",
    "shipping_id", "device_id", "session_id", "request_id", "message_id",
}

MONEY_RE = re.compile(
    r"(?:(?P<cur>\$|€|£|₽)\s?(?P<amount>\d[\d,]*(?:\.\d+)?))|"
    r"(?:(?P<amount2>\d[\d,]*(?:\.\d+)?)\s?(?P<cur2>USD|EUR|GBP|RUB|dollars?|euros?|rubles?))",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\s?(?:AM|PM))?\b", re.IGNORECASE)

MUTATING_VERBS = (
    "book", "cancel", "update", "modify", "change", "edit", "exchange", "return",
    "refund", "send", "transfer", "activate", "deactivate", "suspend", "resume",
    "open", "close", "apply", "submit", "pay", "reimburse", "upgrade", "downgrade",
    "add", "remove", "delete", "create", "issue", "enable", "disable", "reset",
    "unlock", "block", "schedule", "confirm", "complete", "process", "file",
)
_MUTATING_RE = re.compile(r"(?:^|_)(%s)" % "|".join(MUTATING_VERBS), re.IGNORECASE)

AFFIRMATIVE_RE = re.compile(
    r"\b(yes|yeah|yep|sure|okay|ok|confirm(?:ed)?|correct|go ahead|please do|"
    r"proceed|that works|sounds good|do it|i confirm|absolutely|agreed?)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(r"\b(no|nope|don't|do not|cancel that|stop|wait)\b", re.IGNORECASE)

REFUSAL_RE = re.compile(
    r"\b(i (?:can ?not|can't|am unable|'m unable|am not able|'m not able)|"
    r"unable to|not allowed|not permitted|not possible|"
    r"(?:do not|don't) have (?:access|the ability|the capability)|"
    r"no (?:access|ability)|cannot (?:help|assist|do|provide|access|process)|"
    r"i(?:'m| am) (?:not )?(?:authorized|authorised))\b",
    re.IGNORECASE,
)

ACTION_CLAIM_RE = re.compile(
    r"\b(?:i(?:'ve| have)?\s+(?:successfully\s+)?(?:booked|cancelled|canceled|"
    r"updated|modified|changed|sent|transferred|refunded|processed|issued|"
    r"activated|deactivated|suspended|resumed|opened|closed|applied|submitted|"
    r"exchanged|returned|scheduled|reset|unlocked|blocked|paid|added|removed))\b",
    re.IGNORECASE,
)

TOOL_FAILURE_RE = re.compile(
    r'"error"\s*:\s*true|"success"\s*:\s*false|"status"\s*:\s*"(?:error|failed|failure)"'
    r'|^\s*(?:error|failed|not found|invalid|denied)\b'
    r'|\b(?:was|is) not found\b|\bhas failed\b|\bcould not be\b|\bunable to (?:process|complete)\b',
    re.IGNORECASE,
)
SUCCESS_CLAIM_RE = re.compile(
    r"\b(?:successfully|i(?:'ve| have)\s+(?:now\s+)?(?:cancelled|canceled|booked|updated|"
    r"modified|changed|sent|transferred|refunded|processed|issued|activated|deactivated|"
    r"suspended|resumed|opened|closed|applied|submitted|exchanged|returned|paid|added|removed))"
    r"\b|\b(?:has|have) been (?:successfully )?(?:cancelled|canceled|booked|updated|refunded|"
    r"processed|completed|confirmed|issued|sent)\b|\byou(?:'re| are) all set\b",
    re.IGNORECASE,
)

CONFIRMATION_POLICY_RE = re.compile(
    r"(must|need to|required to|have to)[^.\n]{0,120}confirm", re.IGNORECASE
)

KNOWN_FALSE_POSITIVE_CODES = {
    "US", "USA", "AM", "PM", "EST", "PST", "UTC", "GMT", "USD", "EUR", "GBP",
    "RUB", "MMS", "SMS", "SIM", "ID", "API", "FAQ", "JSON", "HTTP", "HTTPS",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_tool_name(name: str) -> str:
    name = (name or "").strip().strip("`")
    name = re.sub(r"^(?:functions?|tools?|default_api|api)\.", "", name, flags=re.I)
    return name


def _context_norm(transcript: Transcript) -> str:
    return norm_token(transcript.raw)


def _squeeze(text: str) -> str:
    return re.sub(r"[\s\-_.]", "", text)


def _present_in_context(
    value: str, context_norm: str, context_squeezed: Optional[str] = None
) -> bool:
    token = norm_token(value)
    if not token:
        return True
    if token in context_norm:
        return True
    # tolerate separators/spaces removed by renderers
    squeezed = _squeeze(token)
    if len(squeezed) >= 5:
        if context_squeezed is None:
            context_squeezed = _squeeze(context_norm)
        return squeezed in context_squeezed
    return False


def _extract_identifiers(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, pattern in ID_PATTERNS:
        for m in pattern.finditer(text or ""):
            value = m.group(0).strip()
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            if kind == "code" and value in KNOWN_FALSE_POSITIVE_CODES:
                continue
            if kind == "phone" and not (10 <= len(re.sub(r"\D", "", value)) <= 15):
                continue
            if kind == "code" and len(re.sub(r"\D", "", value)) < 1:
                continue
            found.append((kind, value))
    return found


def _iter_response_text(response: ResponseTurn) -> Iterable[str]:
    if response.text.strip():
        yield response.text
    for tc in response.tool_calls:
        if not tc.parse_ok and tc.raw_args:
            # The renderer may truncate long JSON with "..."; still audit the
            # visible identifiers.
            yield tc.raw_args
        if tc.args:
            for key, value in tc.args.items():
                if isinstance(value, str):
                    yield f"{key}: {value}"
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            yield item
                        elif isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str):
                                    yield v


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_tool_availability(
    transcript: Transcript, response: ResponseTurn, specs: dict[str, ToolSpec]
) -> list[Signal]:
    if not specs:
        return []
    signals: list[Signal] = []
    context_low = (transcript.raw or "").lower()
    known = {_normalise_tool_name(n).lower() for n in specs}
    # The catalogue may list a tool family without suffixes; accept exact name
    # or a name that differs only by a numeric suffix pattern.
    for tc in response.tool_calls:
        name = _normalise_tool_name(tc.name).lower()
        if name in known:
            continue
        # Discoverable tools carry a numeric suffix (``open_bank_account_4821``);
        # a call without it is still flagged, because the exact name matters.
        base = re.sub(r"_\d{2,6}$", "", name)
        if base in known:
            continue
        # A name that is mentioned somewhere in the context (e.g. in the
        # dialogue or in a catalogue layout we failed to parse) is weaker
        # evidence: downgrade it to a hint for the judge.
        mentioned = name in context_low or (bool(base) and base in context_low)
        signals.append(
            Signal(
                kind="tool_not_available",
                severity="soft" if mentioned else "hard",
                message=(
                    f"the response calls tool '{tc.name}', which is not declared in "
                    "[AVAILABLE TOOLS]"
                ),
                evidence=tc.args_preview(240),
            )
        )
    return signals


def check_tool_arguments(
    response: ResponseTurn, specs: dict[str, ToolSpec]
) -> list[Signal]:
    if not specs:
        return []
    signals: list[Signal] = []
    by_name = {_normalise_tool_name(k).lower(): v for k, v in specs.items()}
    for tc in response.tool_calls:
        spec = by_name.get(_normalise_tool_name(tc.name).lower())
        if spec is None:
            base = re.sub(r"_\d{2,6}$", "", _normalise_tool_name(tc.name).lower())
            spec = by_name.get(base)
        if spec is None or not spec.params:
            continue
        if not tc.parse_ok:
            # Argument JSON may legitimately be truncated by the renderer
            # (``..., ...``); do not report that as a problem.
            if "..." in tc.raw_args or "…" in tc.raw_args:
                continue
            signals.append(
                Signal(
                    kind="unparseable_arguments",
                    severity="soft",
                    message=f"arguments of '{tc.name}' could not be parsed as JSON",
                    evidence=tc.raw_args[:240],
                )
            )
            continue
        # Argument checks are only authoritative for real JSON schemas; a
        # best-effort text parse may mislabel optional fields as required.
        source = getattr(spec, "source", "text")
        # JSON schemas and the official text rendering both carry exact
        # required/enum markers; the legacy heuristics do not.
        severity = "hard" if source in ("json", "text-official") else "soft"
        for issue in validate_tool_call(spec, tc.args):
            signals.append(
                Signal(
                    kind="invalid_tool_arguments",
                    severity=severity,
                    message=f"tool '{tc.name}': {issue}",
                    evidence=tc.args_preview(240),
                )
            )
    return signals


def check_identifier_grounding(
    transcript: Transcript, response: ResponseTurn, severity: str = "hard"
) -> list[Signal]:
    context_norm = _context_norm(transcript)
    # Squeezing is O(len(context)); do it once per row instead of per identifier.
    context_squeezed = _squeeze(context_norm)
    signals: list[Signal] = []
    seen: set[str] = set()

    # Identifier-like tool arguments (reservation_id, order_id, ...).
    for tc in response.tool_calls:
        for key, value in (tc.args or {}).items():
            if not isinstance(value, str) or len(value) < 3:
                continue
            if _ID_ARG_RE.search(key) and not _present_in_context(
                value, context_norm, context_squeezed
            ):
                if value.lower() in seen:
                    continue
                seen.add(value.lower())
                signals.append(
                    Signal(
                        kind="ungrounded_argument",
                        severity=severity,  # type: ignore[arg-type]
                        message=(
                            f"argument '{key}={value}' of tool '{tc.name}' does not "
                            "exist in the provided data"
                        ),
                        evidence=f"{key}={value}",
                    )
                )

    for text in _iter_response_text(response):
        for kind, value in _extract_identifiers(text):
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            if _present_in_context(value, context_norm, context_squeezed):
                continue
            signals.append(
                Signal(
                    kind=f"ungrounded_{kind}",
                    severity=severity,  # type: ignore[arg-type]
                    message=(
                        f"{kind} '{value}' appears in the response but nowhere in the "
                        "provided context"
                    ),
                    evidence=value,
                )
            )
    return signals


def check_numeric_grounding(transcript: Transcript, response: ResponseTurn) -> list[Signal]:
    if not response.text.strip():
        return []
    ctx_numbers = numbers_normalized(transcript.raw)
    ctx_low = transcript.raw.lower()
    unmatched: list[str] = []
    for regex, label in ((MONEY_RE, "amount"), (DATE_RE, "date"), (TIME_RE, "time")):
        for m in regex.finditer(response.text):
            value = m.group(0).strip()
            number = re.search(r"\d[\d,]*(?:\.\d+)?", value)
            if not number:
                continue
            canonical = canonical_number(number.group(0))
            if canonical in ctx_numbers or value.lower() in ctx_low:
                continue
            unmatched.append(f"{label} '{value}'")
    if not unmatched:
        return []
    examples = ", ".join(dict.fromkeys(unmatched))[:300]
    return [
        Signal(
            kind="ungrounded_number",
            severity="soft",
            message=(
                "these concrete values from the response were not found verbatim in the "
                f"context (may be a paraphrase/rounding, or may be invented): {examples}"
            ),
            evidence=examples,
        )
    ]


def check_confirmation(
    transcript: Transcript, response: ResponseTurn
) -> list[Signal]:
    mutating = [
        tc for tc in response.tool_calls if _MUTATING_RE.search(_normalise_tool_name(tc.name))
    ]
    if not mutating:
        return []
    policy = transcript.system_text or ""
    if not CONFIRMATION_POLICY_RE.search(policy):
        return []
    last_user = transcript.last_user_text()
    if AFFIRMATIVE_RE.search(last_user) and not NEGATIVE_RE.search(last_user):
        return []
    names = ", ".join(tc.name for tc in mutating)
    return [
        Signal(
            kind="missing_confirmation",
            severity="soft",
            message=(
                f"the response performs a state-changing action ({names}) but the last "
                "user message does not contain an explicit confirmation; the policy "
                "requires listing the action details and getting a 'yes' first"
            ),
            evidence=last_user[:240],
        )
    ]


def check_false_refusal(transcript: Transcript, response: ResponseTurn) -> list[Signal]:
    text = response.text
    if not text or not REFUSAL_RE.search(text):
        return []
    if not (transcript.tool_calls or transcript.system_text):
        return []
    sentence = ""
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
        if REFUSAL_RE.search(chunk):
            sentence = chunk.strip()
            break
    return [
        Signal(
            kind="possible_false_refusal",
            severity="soft",
            message=(
                "the response refuses or claims inability; check whether the policy and "
                "the available tools actually allow the request"
            ),
            evidence=sentence[:300],
        )
    ]


def check_duplicate_tool_call(
    transcript: Transcript, response: ResponseTurn
) -> list[Signal]:
    previous = [
        (e.tool_call.name, e.tool_call.args_preview(4000))
        for e in transcript.tool_calls
        if e.tool_call
    ]
    if not previous:
        return []
    last_name, last_args = previous[-1]
    for tc in response.tool_calls:
        if tc.name == last_name and tc.args_preview(4000) == last_args:
            return [
                Signal(
                    kind="duplicate_tool_call",
                    severity="soft",
                    message=(
                        f"the response repeats the previous call to '{tc.name}' with "
                        "identical arguments; the result is already in the context"
                    ),
                    evidence=tc.args_preview(200),
                )
            ]
    return []


def check_action_claim(
    transcript: Transcript, response: ResponseTurn
) -> list[Signal]:
    if response.tool_calls or not response.text.strip():
        return []
    m = ACTION_CLAIM_RE.search(response.text)
    if not m:
        return []
    # A prior tool response may already confirm the action; if so this is a
    # legitimate summary rather than a fabricated action.
    recent_tool_text = " ".join(
        e.content.lower() for e in transcript.tool_responses[-3:]
    )
    sentence = ""
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", response.text):
        if ACTION_CLAIM_RE.search(chunk):
            sentence = chunk.strip()
            break
    return [
        Signal(
            kind="claimed_action_without_tool_call",
            severity="soft",
            message=(
                "the response claims that an action was performed, but this turn "
                "contains no tool call; verify against earlier tool results"
            ),
            evidence=(sentence or recent_tool_text)[:300],
        )
    ]


def check_success_after_failure(
    transcript: Transcript, response: ResponseTurn
) -> list[Signal]:
    """The response claims success but the latest tool output reports an error."""
    if not response.text.strip() or not SUCCESS_CLAIM_RE.search(response.text):
        return []
    # Only the latest tool result matters: an error followed by a successful
    # retry is normal and must not be flagged.
    recent = transcript.tool_responses[-1:]
    failed = [e for e in recent if TOOL_FAILURE_RE.search(e.content or "")]
    if not failed:
        return []
    return [
        Signal(
            kind="claimed_success_after_tool_error",
            severity="soft",
            message=(
                "the response claims that an action succeeded, but the most recent tool "
                "result reports an error; check whether the action was actually performed"
            ),
            evidence=(failed[-1].content or "")[:300],
        )
    ]


def check_request_alignment(
    transcript: Transcript, response: ResponseTurn
) -> list[Signal]:
    """The action targets an entity/amount that the user never mentioned."""
    user_messages = [e.content for e in transcript.users if e.content.strip()]
    if not user_messages:
        return []
    users_norm = norm_token(" ".join(user_messages))
    user_ids = {
        value.lower()
        for message in user_messages
        for _, value in _extract_identifiers(message)
    }
    if not user_ids:
        return []
    last_user = user_messages[-1]
    signals: list[Signal] = []

    for tc in response.tool_calls:
        name = _normalise_tool_name(tc.name)
        if not _MUTATING_RE.search(name):
            continue
        arg_ids = {
            str(value)
            for key, value in (tc.args or {}).items()
            if isinstance(value, str)
            and len(value) >= 3
            and _ID_ARG_RE.search(key)
            and key.lower() not in _NON_TARGET_ID_KEYS
            and (key.lower() in _TARGET_ID_KEYS or key.lower().endswith("_id"))
        }
        if not arg_ids:
            continue
        if any(value.lower() in user_ids for value in arg_ids):
            continue
        # The entity may still come from an earlier user message or from the
        # agent's own lookup; only flag when it appears in no user message.
        if all(not _present_in_context(value, users_norm) for value in arg_ids):
            signals.append(
                Signal(
                    kind="action_targets_unrequested_entity",
                    severity="soft",
                    message=(
                        f"tool '{tc.name}' acts on {sorted(arg_ids)} while the user's last "
                        "message mentions different entities; check that this is the "
                        "intended target and not a substituted id"
                    ),
                    evidence=last_user[:240],
                )
            )
    return signals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compute_signals(
    transcript: Transcript,
    response: ResponseTurn,
    specs: dict[str, ToolSpec],
    config: Optional[SignalsConfig] = None,
) -> list[Signal]:
    cfg = config or SignalsConfig()
    signals: list[Signal] = []

    if cfg.enable_tool_availability:
        signals.extend(check_tool_availability(transcript, response, specs))
    if cfg.enable_schema_checks:
        signals.extend(check_tool_arguments(response, specs))
    if cfg.enable_identifier_grounding:
        severity = "hard" if cfg.identifier_severity == "hard" else "soft"
        signals.extend(check_identifier_grounding(transcript, response, severity))
    if cfg.enable_numeric_hints:
        signals.extend(check_numeric_grounding(transcript, response))
    if cfg.enable_confirmation_check:
        signals.extend(check_confirmation(transcript, response))
    if cfg.enable_refusal_check:
        signals.extend(check_false_refusal(transcript, response))
    if cfg.enable_duplicate_check:
        signals.extend(check_duplicate_tool_call(transcript, response))
    if cfg.enable_action_claim_check:
        signals.extend(check_action_claim(transcript, response))
    if cfg.enable_request_alignment:
        signals.extend(check_request_alignment(transcript, response))
    if cfg.enable_success_after_failure:
        signals.extend(check_success_after_failure(transcript, response))

    # Deduplicate by (kind, message).
    uniq: dict[tuple[str, str], Signal] = {}
    for s in signals:
        uniq.setdefault((s.kind, s.message), s)
    ordered = sorted(uniq.values(), key=lambda s: (s.severity != "hard", s.kind))
    soft = [s for s in ordered if s.severity == "soft"][: cfg.max_soft_hints]
    hard = [s for s in ordered if s.severity == "hard"]
    return hard + soft


def has_hard_signal(signals: Iterable[Signal]) -> bool:
    return any(s.severity == "hard" for s in signals)
