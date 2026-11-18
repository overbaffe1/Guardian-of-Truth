"""Keep the portfolio's published snapshots tied to the actual detector."""

import pytest

from docs.showcase.build import BUDGETS, CHECKPOINT, collect, export_matches, outputs


@pytest.fixture(scope="module")
def archive():
    return collect()


def test_demo_variants_cover_three_distinct_cases(archive):
    listed, missing, wrong_id = archive["cases"]
    assert listed["prompt"] == missing["prompt"] == wrong_id["prompt"]
    assert [record["label"] for record in archive["cases"]] == [0, 1, 1]
    assert all(record["label"] == record["expected"] for record in archive["cases"])
    assert not listed["signals"]
    assert ("tool_not_available", "hard") in {(s["kind"], s["severity"]) for s in missing["signals"]}
    assert ("ungrounded_argument", "hard") in {(s["kind"], s["severity"]) for s in wrong_id["signals"]}


def test_packing_frames_report_rendered_payloads(archive):
    scene = archive["packing"]
    assert [cap["budget"] for cap in scene["caps"]] == list(BUDGETS)
    assert [cap["payloads"] for cap in scene["caps"]] == [3, 5, 10, 14]
    assert scene["raw_chars"] == 22_045
    assert scene["caps"][0]["chars"] == 3_789
    for cap in scene["caps"]:
        assert cap["chars"] <= cap["budget"]
        assert cap["proof_present"]
        assert CHECKPOINT in cap["rows"][scene["proof_index"] - 1]["body"]
    assert any(row["state"] == "marker" for row in scene["caps"][0]["rows"])
    assert all(row["state"] == "whole" for row in scene["caps"][-1]["rows"])


def test_scorecard_counts_match_the_labeled_slice(archive):
    scores = archive["scores"]
    assert (scores["total"], scores["positives"]) == (46, 23)
    for mode in scores["regimes"]:
        matrix = mode["matrix"]
        assert matrix["tp"] + matrix["fn"] == scores["positives"]
        assert sum(matrix.values()) == scores["total"]
    _, any_signal, heuristic = scores["regimes"]
    assert any_signal["matrix"]["fp"] == 3
    assert heuristic["matrix"] == {"tp": 12, "fn": 11, "fp": 0, "tn": 23}
    assert heuristic["f1"] == pytest.approx(0.685714)


def test_published_lab_and_plates_are_current(archive):
    stale = [str(path) for path, expected in outputs(archive).items() if not export_matches(path, expected)]
    assert not stale, f"Re-export with python docs/showcase/build.py: {stale}"
