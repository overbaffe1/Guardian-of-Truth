"""Tests for the predict CLI helpers."""

import importlib.util
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
