"""Dataset IO helpers (parquet / csv / jsonl)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from .types import Prediction

logger = logging.getLogger(__name__)

_ID_ALIASES = ("id", "uid", "example_id", "sample_id")
_PROMPT_ALIASES = ("prompt", "context", "input")
_RESPONSE_ALIASES = ("response", "answer", "model_answer", "output")


def _pick(row: dict[str, Any], aliases: Iterable[str], default: Any = "") -> Any:
    for key in aliases:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        # pandas turns missing strings into NaN (a float), which is truthy.
        if isinstance(value, float) and value != value:  # NaN
            continue
        return value
    return default


def read_table(path: str) -> list[dict[str, Any]]:
    """Read a parquet/csv/jsonl file into a list of plain dictionaries."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        df = pd.read_parquet(p)
        records = df.to_dict(orient="records")
    elif suffix in {".jsonl", ".ndjson"}:
        records = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif suffix == ".json":
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("data") or data.get("rows") or list(data.values())
        records = list(data)
    elif suffix in {".csv", ".tsv"}:
        import pandas as pd

        df = pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",", keep_default_na=False)
        records = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    normalised: list[dict[str, Any]] = []
    for i, row in enumerate(records):
        row = dict(row)
        normalised.append(
            {
                # keep the original id value and type: the grader joins on it
                "id": _pick(row, _ID_ALIASES, i),
                "prompt": _pick(row, _PROMPT_ALIASES, ""),
                "response": _pick(row, _RESPONSE_ALIASES, ""),
                "label": row.get("label"),
                "explanation": row.get("explanation"),
                "_raw": row,
            }
        )
    logger.info("Loaded %d rows from %s", len(normalised), path)
    return normalised


def write_predictions(
    path: str, predictions: list[Prediction], with_details: bool = False
) -> None:
    import pandas as pd

    rows = [p.to_row(with_details=with_details) for p in predictions]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["id", "label"] + (["score"] if with_details else []))
    if df["id"].isna().any():
        df["id"] = df["id"].fillna("")
    df["label"] = df["label"].astype(int)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    elif p.suffix.lower() in {".csv", ".tsv"}:
        df.to_csv(p, index=False, sep="\t" if p.suffix.lower() == ".tsv" else ",")
    else:
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    logger.info("Wrote %d predictions to %s", len(df), path)


def dump_debug(path: str, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote debug records to %s", path)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (len(y_true) - fp - fn) / len(y_true) if y_true else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n": float(len(y_true)),
        "positives": float(sum(y_true)),
    }


def best_threshold(
    y_true: list[int], scores: list[float], grid: Optional[list[float]] = None
) -> tuple[float, dict[str, float]]:
    grid = grid or [i / 100 for i in range(1, 100)]
    best = (-1.0, 0.5, {})
    for threshold in grid:
        preds = [1 if s >= threshold else 0 for s in scores]
        metrics = compute_metrics(y_true, preds)
        if metrics["f1"] > best[0]:
            best = (metrics["f1"], threshold, metrics)
    return best[1], best[2]
