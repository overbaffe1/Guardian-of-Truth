import json
import time

import pytest

from guardian.backends import BaseBackend
from guardian.config import RuntimeConfig
from guardian.io_utils import read_table, write_predictions
from guardian.judge import aggregate, heuristic_verdict, parse_verdict
from guardian.pipeline import Detector


class ScriptedBackend(BaseBackend):
    name = "scripted"

    def __init__(self, replies):
        self.replies = replies if isinstance(replies, list) else [replies]
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        out = []
        for request in requests:
            reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            out.append([reply] * max(1, request.n))
        return out


SYSTEM = """⟦SYSTEM⟧
You are an airline agent.
<policy>
Before taking any actions that update the booking database you must obtain explicit
user confirmation (yes).
</policy>
[AVAILABLE TOOLS]
- get_reservation_details: read reservation
- cancel_reservation: cancel reservation
"""

CLEAN_ROW = {
    "id": "clean-1",
    "prompt": SYSTEM
    + "⟦USER⟧\nWhat is the status of EHGLP3?\n⟦ASSISTANT · ход 1⟧\n"
    '\t→ TOOL_CALL get_reservation_details: {"reservation_id": "EHGLP3"}\n'
    '\t← TOOL_RESPONSE get_reservation_details: {"reservation_id": "EHGLP3", "cabin": "economy"}\n',
    "response": "⟦ASSISTANT · ход 2⟧\nYour reservation EHGLP3 is in economy class.",
}

HARD_ROW = {
    "id": "hard-1",
    "prompt": SYSTEM + "⟦USER⟧\nrefund me\n⟦ASSISTANT · ход 1⟧\nok\n",
    "response": '⟦ASSISTANT · ход 2⟧\n\t→ TOOL_CALL issue_full_refund: {"amount": 999}',
}


def make_detector(replies, **overrides):
    config = RuntimeConfig.load(None, overrides or {})
    backend = ScriptedBackend(replies)
    return Detector(config, backend=backend, auto_load_backend=False), backend


def test_hard_rule_short_circuits_judge():
    detector, backend = make_detector(['{"label": 0, "confidence": 0.9}'])
    preds = detector.predict_rows([HARD_ROW])
    assert preds[0].label == 1
    assert backend.requests == []  # judge never called


def test_clean_row_uses_judge_label_zero():
    detector, _ = make_detector(
        ['{"label": 0, "confidence": 0.9, "reason": "grounded"}']
    )
    preds = detector.predict_rows([CLEAN_ROW])
    assert preds[0].label == 0
    assert preds[0].judge_label == 0


def test_judge_positive_is_used():
    detector, _ = make_detector(
        ['{"label": 1, "confidence": 0.8, "hallucination_type": "unsupported"}']
    )
    preds = detector.predict_rows([CLEAN_ROW])
    assert preds[0].label == 1
    assert preds[0].score >= 0.5


def test_unparseable_reply_falls_back_to_rules():
    detector, _ = make_detector(["I think it is fine."])
    preds = detector.predict_rows([CLEAN_ROW])
    assert preds[0].label == 0
    assert preds[0].signals == []


def test_majority_vote():
    replies = [
        json.dumps({"label": 1, "confidence": 0.7}),
        json.dumps({"label": 0, "confidence": 0.7}),
        json.dumps({"label": 1, "confidence": 0.6}),
    ]
    detector, _ = make_detector(replies, judge={"self_consistency": 3})
    preds = detector.predict_rows([CLEAN_ROW])
    assert preds[0].label == 1
    assert preds[0].judge_confidence is not None


def test_deadline_falls_back_to_heuristics():
    detector, backend = make_detector(['{"label": 1, "confidence": 0.9}'])
    preds = detector.predict_rows([CLEAN_ROW], deadline=time.monotonic() - 1)
    assert preds[0].label == 0
    assert backend.requests == []


def test_predictions_roundtrip_parquet(tmp_path):
    detector, _ = make_detector(
        ['{"label": 0, "confidence": 0.9}']
    )
    preds = detector.predict_rows([CLEAN_ROW, HARD_ROW])
    out = tmp_path / "preds.parquet"
    write_predictions(str(out), preds, with_details=True)
    rows = read_table(str(out))
    assert [r["id"] for r in rows] == ["clean-1", "hard-1"]
    assert [int(r["label"]) for r in rows] == [0, 1]


def test_parse_verdict_variants():
    assert parse_verdict('{"label": 1, "confidence": 0.9}').label == 1
    assert parse_verdict('```json\n{"label": 0, "confidence": 90}\n```').label == 0
    assert parse_verdict('{"label": 0, "confidence": 90}').confidence == pytest.approx(0.9)
    assert parse_verdict('{"verdict": "yes", "confidence": 0.4}').label == 1
    assert parse_verdict("garbage").failed


def test_aggregate_single_and_multi():
    single = [parse_verdict('{"label": 1, "confidence": 0.8}')]
    assert aggregate(single).label == 1
    multi = [
        parse_verdict('{"label": 1, "confidence": 0.8}'),
        parse_verdict('{"label": 0, "confidence": 0.8}'),
        parse_verdict('{"label": 1, "confidence": 0.9}'),
    ]
    result = aggregate(multi)
    assert result.label == 1
    assert result.votes == [1, 0, 1]


def test_heuristic_judge_prefers_hard_signals():
    from guardian.types import Signal

    verdict = heuristic_verdict([Signal("tool_not_available", "hard", "x")])
    assert verdict.label == 1


def test_json_extraction_prefers_verdict_object():
    from guardian.judge import parse_verdict

    text = (
        "The claim {amount} looks suspicious, but {x} is not real. "
        'Final: {"claims": [], "label": 1, "confidence": 0.7}'
    )
    result = parse_verdict(text)
    assert result.label == 1
    assert not result.failed


def test_ids_preserve_input_dtype(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "id": [10, 20],
            "prompt": [SYSTEM + "⟦USER⟧\nhello\n"] * 2,
            "response": ["⟦ASSISTANT · ход 1⟧\nHello, how can I help?"] * 2,
        }
    )
    src = tmp_path / "in.parquet"
    df.to_parquet(src, index=False)

    detector = Detector(RuntimeConfig.load(), backend=None, auto_load_backend=False)
    predictions = detector.predict_rows(read_table(str(src)))
    assert [p.id for p in predictions] == [10, 20]

    out = tmp_path / "out.parquet"
    write_predictions(str(out), predictions)
    back = pd.read_parquet(out)
    assert back["id"].tolist() == [10, 20]
    assert list(back.columns) == ["id", "label"]


class VerifyBackend(BaseBackend):
    """First pass says 0, second opinion says 1."""

    name = "verify"

    def __init__(self):
        self.systems = []

    def generate(self, requests):
        out = []
        for request in requests:
            system = request.messages[0]["content"]
            self.systems.append(system)
            if "SECOND, independent review" in system:
                out.append(['{"label": 1, "confidence": 0.9, "claims": '
                            '[{"claim": "x", "status": "unsupported"}]}'])
            else:
                out.append(['{"label": 0, "confidence": 0.9}'])
        return out


def test_pipeline_uses_second_opinion():
    row = dict(CLEAN_ROW)
    row["id"] = "verify-1"
    backend = VerifyBackend()
    detector = Detector(RuntimeConfig.load(), backend=backend, auto_load_backend=False)
    predictions = detector.predict_rows([row])
    assert predictions[0].label == 1
    assert detector.stats["verified"] == 1
    assert detector.stats["verify_overrides"] == 1
    assert any("SECOND, independent review" in s for s in backend.systems)


def test_pipeline_second_opinion_can_be_disabled():
    row = dict(CLEAN_ROW)
    row["id"] = "verify-2"
    backend = VerifyBackend()
    detector = Detector(
        RuntimeConfig.load(overrides={"judge": {"verify_negatives": False}}),
        backend=backend,
        auto_load_backend=False,
    )
    predictions = detector.predict_rows([row])
    assert predictions[0].label == 0
    assert detector.stats["verified"] == 0
