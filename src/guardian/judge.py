"""LLM verdict parsing plus a deterministic fallback judge."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from .backends import BaseBackend, GenerationRequest
from .types import JudgeResult, Signal

logger = logging.getLogger(__name__)

_LABEL_RE = re.compile(r'"?label"?\s*[:=]\s*"?([01]|true|false|yes|no)"?', re.IGNORECASE)
_CONF_RE = re.compile(r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
_TYPE_RE = re.compile(r'"?hallucination_type"?\s*[:=]\s*"([a-z_]+)"', re.IGNORECASE)
_REASON_RE = re.compile(r'"?reason"?\s*[:=]\s*"((?:[^"\\]|\\.){0,400})"', re.IGNORECASE)


_VERDICT_KEYS = ("label", "verdict", "is_hallucination", "hallucination", "result")


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Return the most relevant JSON object from a model reply.

    Models sometimes emit reasoning with braces before the verdict, so objects
    that actually contain a verdict key are preferred (the last one wins).
    """
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        if any(key in value for key in _VERDICT_KEYS):
            candidates.append(value)
        elif not candidates:
            candidates.append(value)
    return candidates[-1] if candidates else None


_BAD_CLAIM_STATUSES = {
    "unsupported",
    "contradicted",
    "action_error",
    "policy_violation",
    "ungrounded",
    "not_supported",
    "hallucination",
    "false",
}


def bad_claim_count(text: str) -> Optional[int]:
    """Number of claims marked as problematic in a judge reply.

    Returns ``None`` when the reply has no parsable ``claims`` list (older or
    shorter formats), so callers can decide whether the check applies.
    """
    payload = _extract_json_object(text or "")
    if payload is None:
        return None
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return None
    count = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        status = str(claim.get("status", "")).strip().lower().replace(" ", "_")
        if status in _BAD_CLAIM_STATUSES:
            count += 1
    return count


def _coerce_label(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0 if int(value) == 0 else None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "hallucination", "есть"}:
            return 1
        if low in {"0", "false", "no", "none", "нет"}:
            return 0
    return None


def _coerce_confidence(value: Any, default: float = 0.6) -> float:
    try:
        conf = float(value)
    except Exception:
        return default
    if conf > 1.0 and conf <= 100.0:
        conf /= 100.0
    return max(0.0, min(1.0, conf))


def parse_verdict(text: str) -> JudgeResult:
    """Parse a model reply into a :class:`JudgeResult` (never raises)."""
    payload = _extract_json_object(text or "")
    if payload is not None:
        label = _coerce_label(payload.get("label"))
        if label is None:
            # Some models put the verdict in "verdict" / "is_hallucination".
            for key in ("verdict", "is_hallucination", "hallucination", "result"):
                if key in payload:
                    label = _coerce_label(payload[key])
                    if label is not None:
                        break
        if label is not None:
            return JudgeResult(
                label=label,
                confidence=_coerce_confidence(payload.get("confidence", 0.7)),
                hallucination_type=str(payload.get("hallucination_type", "none"))[:40],
                reason=str(payload.get("reason", ""))[:600],
                raw=(text or "")[:4000],
                votes=[label],
            )

    match = _LABEL_RE.search(text or "")
    if match:
        label = _coerce_label(match.group(1))
        if label is not None:
            conf_match = _CONF_RE.search(text or "")
            type_match = _TYPE_RE.search(text or "")
            reason_match = _REASON_RE.search(text or "")
            return JudgeResult(
                label=label,
                confidence=_coerce_confidence(conf_match.group(1)) if conf_match else 0.55,
                hallucination_type=type_match.group(1) if type_match else "unknown",
                reason=(reason_match.group(1) if reason_match else "")[:600],
                raw=(text or "")[:4000],
                votes=[label],
            )
    return JudgeResult(
        label=0,
        confidence=0.0,
        hallucination_type="unknown",
        reason="unparseable judge output",
        raw=(text or "")[:4000],
        votes=[],
        failed=True,
    )


def aggregate(runs: list[JudgeResult]) -> JudgeResult:
    """Majority vote / mean confidence over self-consistency samples."""
    parsed = [r for r in runs if not r.failed]
    if not parsed:
        return runs[0] if runs else JudgeResult(label=0, confidence=0.0, failed=True)
    votes = [r.label for r in parsed]
    p1 = sum(votes) / len(votes)
    label = 1 if p1 >= 0.5 else 0
    if len(votes) == 1:
        confidence = parsed[0].confidence
    else:
        agree = sum(1 for v in votes if v == label) / len(votes)
        same = [r.confidence for r in parsed if r.label == label]
        mean_conf = sum(same) / len(same) if same else 0.5
        confidence = 0.6 * agree + 0.4 * mean_conf
    best = max(parsed, key=lambda r: r.confidence)
    return JudgeResult(
        label=label,
        confidence=max(0.0, min(1.0, confidence)),
        hallucination_type=best.hallucination_type,
        reason=best.reason,
        raw=best.raw,
        votes=votes,
    )


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

#: Weight of each soft hint when no LLM verdict is available. Re-fitted on the
#: real labelled slice (see reports/valid-rules.json): a weight at or above the
#: fire score below makes that single hint decide the row on its own, so the
#: weights track the measured per-signal precision.
SOFT_WEIGHTS = {
    # precision 1.00 (n=6): the catalogue is parsed from the very block the
    # agent saw, so an undeclared tool is a reliable error.
    "tool_not_available": 0.9,
    "placeholder_argument": 0.9,
    # precision 1.00 (n=1-2) on real labels: invented or mis-targeted entities
    # change what the user sees.
    "ungrounded_email": 1.4,
    "ungrounded_phone": 1.4,
    "ungrounded_doc_id": 1.5,
    "ungrounded_order_id": 1.4,
    "ungrounded_code": 1.0,
    "ungrounded_argument": 1.3,
    "action_targets_unrequested_entity": 0.85,
    "claimed_action_without_tool_call": 0.85,
    "duplicate_tool_call": 0.85,
    "missing_confirmation": 0.85,
    "claimed_success_after_tool_error": 0.9,
    # Deliberately weak: both fired on true negatives in the real slice
    # (precision 0.00 and 0.33). They stay as judge hints, never decisive.
    "ungrounded_number": 0.35,
    "possible_false_refusal": 0.35,
    "unparseable_arguments": 0.2,
}


def heuristic_verdict(signals: list[Signal], response_empty: bool = False) -> JudgeResult:
    """Label from deterministic hints only.

    Used when no model is available (dry runs, CPU-only environments) and as
    a safety net if the backend crashes.  It is intentionally conservative.
    """
    if response_empty:
        return JudgeResult(label=0, confidence=0.8, hallucination_type="none",
                           reason="empty response", votes=[0])
    hard = [s for s in signals if s.severity == "hard"]
    if hard:
        return JudgeResult(
            label=1,
            confidence=0.9,
            hallucination_type="tool_error",
            reason=hard[0].message,
            votes=[1],
        )
    score = sum(SOFT_WEIGHTS.get(s.kind, 0.3) for s in signals if s.severity == "soft")
    # 0.8 was tuned on the synthetic tau2-based development set: it keeps
    # precision high while recovering single strong hints (e.g. an invented
    # monetary amount).
    p1 = 1.0 / (1.0 + math.exp(-(score - 0.8) * 1.8))
    label = 1 if p1 >= 0.5 else 0
    reason = "; ".join(s.kind for s in signals[:4]) or "no suspicious signals"
    # ``confidence`` is the confidence in the predicted label, matching the
    # convention used for the LLM verdicts.
    return JudgeResult(
        label=label,
        confidence=p1 if label == 1 else 1.0 - p1,
        hallucination_type="unsupported" if label else "none",
        reason=reason,
        votes=[label],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class JudgeItem:
    uid: str
    messages: list[dict[str, str]]
    signals: list[Signal]
    #: second-opinion prompt (used when the first pass answers 0)
    verify_messages: Optional[list[dict[str, str]]] = None
    #: optional factory ``rebuild(scale) -> messages`` used to shrink the
    #: context when the run is projected to exceed the time budget
    rebuild: Optional[Any] = None


class JudgeRunner:
    def __init__(self, backend: Optional[BaseBackend], cfg: dict[str, Any]):
        self.backend = backend
        self.cfg = cfg
        self.failed_batches = 0
        #: diagnostics for the second-opinion pass
        self.verified = 0
        self.verify_overrides = 0

    # ------------------------------------------------------------------ verify
    def _verify_negatives(
        self,
        items: list[JudgeItem],
        results: dict[str, JudgeResult],
        deadline: Optional[float],
    ) -> None:
        import time

        threshold = float(self.cfg.get("verify_threshold", 0.65))
        max_tokens = int(self.cfg.get("verify_max_tokens", self.cfg.get("max_tokens", 448)))
        batch_size = max(1, int(self.cfg.get("batch_size", 16)))
        for start in range(0, len(items), batch_size):
            if deadline is not None and time.monotonic() > deadline:
                logger.warning(
                    "Deadline reached, skipping the second-opinion pass for %d items",
                    len(items) - start,
                )
                return
            batch = items[start : start + batch_size]
            requests = [
                GenerationRequest(
                    messages=item.verify_messages or item.messages,
                    max_tokens=max_tokens,
                    temperature=float(self.cfg.get("verify_temperature", 0.0)),
                    n=1,
                )
                for item in batch
            ]
            try:
                outputs = self.backend.generate(requests)  # type: ignore[union-attr]
            except Exception as exc:
                self.failed_batches += 1
                logger.error("Second-opinion batch failed (%s); skipping it", exc)
                continue
            for item, texts in zip(batch, outputs):
                if not texts:
                    continue
                verdict = parse_verdict(texts[0])
                self.verified += 1
                if verdict.failed or verdict.label != 1 or verdict.confidence < threshold:
                    continue
                # Guard against a "1" that contradicts its own claim list.
                if bad_claim_count(texts[0]) == 0:
                    logger.debug("Second opinion said 1 but listed no bad claim; ignored")
                    continue
                previous = results.get(item.uid)
                if previous is None or previous.label != 0:
                    continue
                self.verify_overrides += 1
                results[item.uid] = JudgeResult(
                    label=1,
                    confidence=max(verdict.confidence, previous.confidence),
                    hallucination_type=verdict.hallucination_type or previous.hallucination_type,
                    reason=("second opinion: " + (verdict.reason or ""))[:600],
                    raw=verdict.raw,
                    votes=[1],
                )

    def run(self, items: list[JudgeItem], deadline: Optional[float] = None) -> dict[str, JudgeResult]:
        if not items:
            return {}
        if self.backend is None:
            return {item.uid: heuristic_verdict(item.signals) for item in items}

        import time

        n = max(1, int(self.cfg.get("self_consistency", 1)))
        temperature = float(self.cfg.get("temperature", 0.0))
        if n > 1:
            temperature = max(temperature, float(self.cfg.get("consistency_temperature", 0.4)))

        results: dict[str, JudgeResult] = {}
        batch_size = max(1, int(self.cfg.get("batch_size", 16)))
        scale = 1.0
        run_started = time.monotonic()
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            if deadline is not None and start > 0:
                now = time.monotonic()
                elapsed = max(1e-6, now - run_started)
                rate = start / elapsed
                projected = (len(items) - start) / max(rate, 1e-6)
                available = deadline - now
                if projected > available * 0.95 and scale > 0.4:
                    scale = max(0.4, scale * 0.7)
                    logger.warning(
                        "Time budget: projected %.0fs > available %.0fs, "
                        "shrinking context to %.0f%% for the next batch",
                        projected, available, scale * 100,
                    )
                    for item in batch:
                        if item.rebuild is not None:
                            try:
                                item.messages = item.rebuild(scale)
                            except Exception as exc:  # pragma: no cover
                                logger.warning("Rebuild failed: %s", exc)
            if deadline is not None and time.monotonic() > deadline:
                logger.warning("Deadline reached, judging remaining items with heuristics")
                for item in items[start:]:
                    results[item.uid] = heuristic_verdict(item.signals)
                break
            requests = [
                GenerationRequest(
                    messages=item.messages,
                    max_tokens=int(self.cfg.get("max_tokens", 640)),
                    temperature=temperature,
                    n=n,
                )
                for item in batch
            ]
            try:
                outputs = self.backend.generate(requests)
            except Exception as exc:
                self.failed_batches += 1
                logger.error("Judge batch failed (%s); retrying one by one", exc)
                outputs = []
                for request in requests:
                    try:
                        outputs.append(self.backend.generate([request])[0])
                    except Exception as exc2:
                        logger.error("Judge request failed: %s", exc2)
                        outputs.append([])
            for item, texts in zip(batch, outputs):
                if not texts:
                    results[item.uid] = heuristic_verdict(item.signals)
                    continue
                runs = [parse_verdict(t) for t in texts]
                if all(r.failed for r in runs) and self.cfg.get("retry_on_parse_error", True):
                    try:
                        retry = self.backend.generate(
                            [
                                GenerationRequest(
                                    messages=item.messages
                                    + [
                                        {
                                            "role": "assistant",
                                            "content": (texts[0] or "")[:800],
                                        },
                                        {
                                            "role": "user",
                                            "content": (
                                                "Your previous answer was not valid JSON. "
                                                "Return ONLY the JSON object, nothing else."
                                            ),
                                        },
                                    ],
                                    max_tokens=int(self.cfg.get("max_tokens", 640)),
                                    temperature=0.0,
                                    n=1,
                                )
                            ]
                        )[0]
                        runs = [parse_verdict(t) for t in retry]
                    except Exception:
                        pass
                results[item.uid] = aggregate(runs)

        if self.cfg.get("verify_negatives", True):
            # Give up on the second pass if less than ~10% of the deadline is left.
            if deadline is None or time.monotonic() < deadline - max(15.0, 0.05 * (deadline - run_started)):
                pending = [
                    item
                    for item in items
                    if item.verify_messages
                    and item.uid in results
                    and results[item.uid].label == 0
                    and not results[item.uid].failed
                ]
                if pending:
                    logger.info("Second opinion on %d responses cleared by the first pass", len(pending))
                    self._verify_negatives(pending, results, deadline)
        return results
