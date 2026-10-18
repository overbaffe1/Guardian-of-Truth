"""Tests for the judge runner, verdict parsing and the second-opinion pass."""

from guardian.judge import (
    JudgeItem,
    JudgeRunner,
    heuristic_verdict,
    parse_verdict,
)
from guardian.types import Signal


class ScriptedBackend:
    """Returns a fixed reply for the first pass and another one for the second."""

    name = "scripted"

    def __init__(self, first: str, second: str = ""):
        self.first = first
        self.second = second
        self.calls = []

    def generate(self, requests):
        outputs = []
        for request in requests:
            system = request.messages[0]["content"]
            self.calls.append(system)
            if "SECOND, independent review" in system:
                outputs.append([self.second])
            else:
                outputs.append([self.first])
        return outputs


def _item(uid: str, verify: bool = True, signals=None) -> JudgeItem:
    messages = [
        {"role": "system", "content": "You are a meticulous auditor."},
        {"role": "user", "content": f"content for {uid}"},
    ]
    verify_messages = None
    if verify:
        verify_messages = [
            {"role": "system", "content": "This is a SECOND, independent review."},
            {"role": "user", "content": f"content for {uid}"},
        ]
    return JudgeItem(uid=uid, messages=messages, signals=signals or [], verify_messages=verify_messages)


def test_parse_verdict_plain_json():
    result = parse_verdict('{"label": 1, "confidence": 0.8, "reason": "x"}')
    assert result.label == 1
    assert not result.failed
    assert result.votes == [1]


def test_parse_verdict_russian_words():
    result = parse_verdict('{"label": "нет", "confidence": 80}')
    assert result.label == 0
    assert result.confidence == 0.8


def test_parse_verdict_garbage_is_failed():
    result = parse_verdict("I think it is probably fine.")
    assert result.failed
    assert result.label == 0


def test_second_opinion_overrides_negative():
    backend = ScriptedBackend(
        first='{"label": 0, "confidence": 0.9, "reason": "looks fine"}',
        second='{"label": 1, "confidence": 0.9, "reason": "amount is invented"}',
    )
    runner = JudgeRunner(backend, {"verify_negatives": True, "batch_size": 4})
    results = runner.run([_item("a")])
    assert results["a"].label == 1
    assert runner.verify_overrides == 1
    assert len(backend.calls) == 2


def test_second_opinion_low_confidence_is_ignored():
    backend = ScriptedBackend(
        first='{"label": 0, "confidence": 0.9}',
        second='{"label": 1, "confidence": 0.3}',
    )
    runner = JudgeRunner(backend, {"verify_negatives": True, "verify_threshold": 0.65})
    results = runner.run([_item("a")])
    assert results["a"].label == 0
    assert runner.verify_overrides == 0


def test_second_opinion_skipped_for_positives():
    backend = ScriptedBackend(
        first='{"label": 1, "confidence": 0.9}',
        second='{"label": 0, "confidence": 0.9}',
    )
    runner = JudgeRunner(backend, {"verify_negatives": True})
    results = runner.run([_item("a")])
    assert results["a"].label == 1
    assert len(backend.calls) == 1


def test_second_opinion_can_be_disabled():
    backend = ScriptedBackend(
        first='{"label": 0, "confidence": 0.9}',
        second='{"label": 1, "confidence": 0.9}',
    )
    runner = JudgeRunner(backend, {"verify_negatives": False})
    results = runner.run([_item("a")])
    assert results["a"].label == 0
    assert len(backend.calls) == 1


def test_no_backend_falls_back_to_heuristics():
    signal = Signal(kind="ungrounded_number", severity="hard", message="invented 250 USD")
    runner = JudgeRunner(None, {})
    results = runner.run([_item("a", verify=False, signals=[signal])])
    assert results["a"].label == 1
    assert results["a"].votes == [1]


def test_heuristic_confidence_is_label_consistent():
    positive = heuristic_verdict(
        [Signal(kind="ungrounded_number", severity="soft", message="m")]
    )
    assert positive.label == 1
    assert positive.confidence > 0.5
    clean = heuristic_verdict([], response_empty=True)
    assert clean.label == 0
    assert clean.confidence > 0.5
