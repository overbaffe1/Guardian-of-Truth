"""Tests for the command-line tools and the README demo."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_predict():
    spec = importlib.util.spec_from_file_location("predict_cli", ROOT / "scripts" / "predict.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_coerce_types():
    predict = _load_predict()
    assert predict._coerce("false") is False
    assert predict._coerce("TRUE") is True
    assert predict._coerce("0.65") == 0.65
    assert predict._coerce("12") == 12
    assert predict._coerce("strict") == "strict"
    assert predict._coerce("null") is None


def test_assign_dotted_path():
    predict = _load_predict()
    target: dict = {}
    predict._assign(target, "judge.verify_negatives", False)
    predict._assign(target, "judge.verify_threshold", 0.7)
    predict._assign(target, "backend", "vllm")
    assert target == {
        "judge": {"verify_negatives": False, "verify_threshold": 0.7},
        "backend": "vllm",
    }


def test_blend_vote(tmp_path):
    import pandas as pd

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    out = tmp_path / "out.parquet"
    pd.DataFrame({"id": ["1", "2", "3"], "label": [1, 0, 1]}).to_parquet(a, index=False)
    pd.DataFrame({"id": ["1", "2", "3"], "label": [0, 0, 1]}).to_parquet(b, index=False)

    spec = importlib.util.spec_from_file_location("blend_cli", ROOT / "scripts" / "blend.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main(["--preds", str(a), str(b), "--out", str(out)]) == 0

    result = pd.read_parquet(out)
    assert list(result.columns) == ["id", "label"]
    assert result["label"].tolist() == [0, 0, 1]


def test_doctor_smoke_on_heuristic(tmp_path, capsys):
    import pandas as pd

    src = tmp_path / "in.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "prompt": [
                "⟦SYSTEM⟧\nAgent.\n<policy>\nBe nice.\n</policy>\n⟦USER⟧\nhi\n",
                "⟦SYSTEM⟧\nAgent.\n<policy>\nBe nice.\n</policy>\n⟦USER⟧\nbye\n",
            ],
            "response": ["⟦ASSISTANT · ход 1⟧\nHello!", "⟦ASSISTANT · ход 1⟧\nGoodbye!"],
        }
    ).to_parquet(src, index=False)

    spec = importlib.util.spec_from_file_location("doctor_cli", ROOT / "scripts" / "doctor.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    code = module.main(["--input", str(src), "--limit", "2", "--backend", "heuristic"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"rows": 2' in out


def test_portfolio_demo_runs_without_model(tmp_path):
    import pandas as pd

    fixture = ROOT / "tests/fixtures/portfolio_demo.jsonl"
    output = tmp_path / "predictions.parquet"
    debug = tmp_path / "debug.jsonl"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/predict.py"),
            "--backend", "heuristic", "--input", str(fixture), "--output", str(output),
            "--debug-jsonl", str(debug),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["backend"] == "heuristic"

    expected = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    frame = pd.read_parquet(output)
    assert list(frame.columns) == ["id", "label"]
    assert frame.to_dict(orient="records") == [
        {"id": row["id"], "label": row["label"]} for row in expected
    ]
    clean, flagged = (json.loads(line) for line in debug.read_text(encoding="utf-8").splitlines())
    assert clean["signals"] == []
    assert any(
        signal["kind"] == "tool_not_available" and signal["severity"] == "hard"
        for signal in flagged["signals"]
    )
