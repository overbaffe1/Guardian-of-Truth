"""Combining deterministic signals with the LLM verdict."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .judge import SOFT_WEIGHTS, heuristic_verdict
from .types import JudgeResult, Prediction, Signal


@dataclass
class FusionConfig:
    threshold: float = 0.5
    rule_mode: str = "strict"  # strict | hint
    soft_signal_override: float = 0.0
    min_confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FusionConfig":
        data = data or {}
        return cls(
            threshold=float(data.get("threshold", 0.5)),
            rule_mode=str(data.get("rule_mode", "strict")),
            soft_signal_override=float(data.get("soft_signal_override", 0.0)),
            min_confidence=float(data.get("min_confidence", 0.0)),
        )


def _soft_score(signals: list[Signal]) -> float:
    return sum(SOFT_WEIGHTS.get(s.kind, 0.3) for s in signals if s.severity == "soft")


def fuse(
    uid: str,
    signals: list[Signal],
    judge: Optional[JudgeResult],
    cfg: FusionConfig,
    backend_name: str = "",
) -> Prediction:
    hard = [s for s in signals if s.severity == "hard"]

    if hard and cfg.rule_mode == "strict":
        return Prediction(
            id=uid,
            label=1,
            score=1.0,
            judge_label=judge.label if judge else None,
            judge_confidence=judge.confidence if judge else None,
            signals=signals,
            reason=hard[0].message,
            backend="rules" if not backend_name else backend_name,
        )

    if judge is None or judge.failed:
        judge = heuristic_verdict(signals)

    if judge.votes and len(judge.votes) > 1:
        p1 = sum(judge.votes) / len(judge.votes)
    else:
        p1 = judge.confidence if judge.label == 1 else 1.0 - judge.confidence
    p1 = max(0.0, min(1.0, p1))

    if judge.confidence < cfg.min_confidence and not hard:
        fallback = heuristic_verdict(signals)
        p1 = fallback.confidence if fallback.label == 1 else 1.0 - fallback.confidence
        judge = fallback

    label = 1 if p1 >= cfg.threshold else 0

    # Optional: strong accumulation of soft hints can override a "0".
    if (
        label == 0
        and cfg.soft_signal_override > 0
        and _soft_score(signals) >= cfg.soft_signal_override
    ):
        label = 1
        p1 = max(p1, min(0.99, 1.0 / (1.0 + math.exp(-(_soft_score(signals) - 1.0)))))

    if hard and cfg.rule_mode != "strict":
        # In hint mode hard signals only nudge the score.
        p1 = max(p1, 0.75)
        label = 1 if p1 >= cfg.threshold else 0

    return Prediction(
        id=uid,
        label=label,
        score=p1,
        judge_label=judge.label,
        judge_confidence=judge.confidence,
        signals=signals,
        reason=judge.reason,
        backend=backend_name or ("rules" if hard else "heuristic"),
    )
