#!/usr/bin/env python3
"""Export the Guardian field archive from the CPU detector (no model or network)."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE.parent / "assets" / "showcase"
sys.path.insert(0, str(ROOT / "src"))

from guardian.config import RuntimeConfig  # noqa: E402
from guardian.evidence import build_evidence  # noqa: E402
from guardian.io_utils import compute_metrics, read_table  # noqa: E402
from guardian.pipeline import Detector  # noqa: E402
from guardian.toolspec import extract_available_tools  # noqa: E402
from guardian.transcript import parse_response, parse_transcript  # noqa: E402

ORDER = "#W1042087"
CHECKPOINT = "D-17"
PROOF_INDEX = 4
BUDGETS = (4_000, 8_000, 16_000, 30_000)
LABELS = {"hard": "Только hard", "any": "Любой сигнал", "cpu": "Эвристика CPU"}


def export_cases(detector: Detector) -> list[dict]:
    listed, invented = read_table(str(ROOT / "tests/fixtures/portfolio_demo.jsonl"))
    wrong_id = {**listed, "id": "invented-id", "label": 1,
                "response": listed["response"].replace(ORDER, "#W1991420")}
    rows = [listed, invented, wrong_id]
    titles = ["Вызов из каталога", "Выдуманный инструмент", "Чужой номер заказа"]
    notes = [
        "Имя есть в каталоге, а номер заказа — в реплике пользователя. Правила не нашли ошибки.",
        "В каталоге только lookup_order. Новый refund_order даёт жёсткий сигнал, судья не нужен.",
        "Инструмент настоящий, но #W1991420 не встречается в контексте. Правило проверяет не только имя вызова.",
    ]
    predictions = detector.predict_rows(rows)
    return [
        {
            "id": row["id"], "title": title, "note": note, "prompt": row["prompt"],
            "response": row["response"], "expected": row["label"], "label": pred.label,
            "signals": [{"kind": s.kind, "severity": s.severity, "message": s.message} for s in pred.signals],
        }
        for row, title, note, pred in zip(rows, titles, notes, predictions)
    ]


def pack_scene(config: RuntimeConfig) -> dict:
    """Build an invented shipment log. No competition text appears in the lab."""
    header = (
        "⟦SYSTEM⟧\nYou are a dispatch assistant.\n"
        "<policy>\nReport only recorded checkpoints. Do not invent a delivery date.\n</policy>\n"
        "[AVAILABLE TOOLS]\n- lookup_order — Read archived shipment status\n"
        "    order_id: string! — Order identifier\n\n"
        f"⟦USER⟧\nГде сейчас заказ {ORDER}?\n"
    )
    raw = [header]
    entries = []
    for index in range(1, 15):
        relevant = index == PROOF_INDEX
        order = ORDER if relevant else f"#W107{index:04}"
        tool = f"lookup_order_{index:02}"
        body = json.dumps({
            "order_id": order,
            "checkpoint": CHECKPOINT if relevant else f"R-{index:02}",
            "status": "in_transit" if relevant else "queued",
            "note": (f"Archive carrier lot {index:02}; sealed crate in dispatch ledger. " * (17 + index)).strip(),
        }, ensure_ascii=False, separators=(",", ":"))
        raw.append(f'→ TOOL_CALL {tool}: {{"order_id":"{order}"}}\n← TOOL_RESPONSE {tool}: {body}\n')
        entries.append({"number": index, "tool": tool, "body": body, "proof": relevant})
    prompt = "".join(raw).rstrip()
    reply = f"Заказ {ORDER} в статусе in_transit. Контрольная точка: {CHECKPOINT}."
    transcript = parse_transcript(prompt)
    response = parse_response(reply)
    specs = extract_available_tools(transcript.system_text)
    caps = []
    for budget in BUDGETS:
        evidence = build_evidence(transcript, response, specs, replace(config.evidence_config(), max_chars=budget))
        visible = {event.tool_name: event.content for event in parse_transcript(evidence.text).tool_responses}
        rows = []
        for entry in entries:
            body = visible.get(entry["tool"], "")
            if not body:
                state = "gone"
            elif body == "[omitted to fit context]":
                state = "marker"
            elif body == entry["body"]:
                state = "whole"
            else:
                state = "short"
            rows.append({"state": state, "body": body})
        proof = rows[PROOF_INDEX - 1]["body"]
        caps.append({
            "budget": budget, "chars": len(evidence.text), "rows": rows,
            "payloads": sum(row["state"] in {"short", "whole"} for row in rows),
            "proof_present": all(token in proof for token in (ORDER, CHECKPOINT, "in_transit")),
        })
    return {"raw_chars": len(prompt), "reply": reply, "proof_index": PROOF_INDEX, "entries": entries, "caps": caps}


def scores(detector: Detector) -> dict:
    rows = read_table(str(ROOT / "valid.parquet"))
    truth = [int(row["label"]) for row in rows]
    predicted = detector.predict_rows(rows)
    variants = {
        "hard": [int(any(s.severity == "hard" for s in pred.signals)) for pred in predicted],
        "any": [int(bool(pred.signals)) for pred in predicted],
        "cpu": [pred.label for pred in predicted],
    }
    regimes = []
    for key, labels in variants.items():
        matrix = dict.fromkeys(("tp", "fn", "fp", "tn"), 0)
        for expected, actual in zip(truth, labels):
            name = ("tp" if actual else "fn") if expected else ("fp" if actual else "tn")
            matrix[name] += 1
        metrics = compute_metrics(truth, labels)
        regimes.append({
            "key": key, "title": LABELS[key], "matrix": matrix,
            **{k: round(metrics[k], 6) for k in ("precision", "recall", "f1")},
        })
    return {"total": len(truth), "positives": sum(truth), "regimes": regimes}


def score_csv(data: dict) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["mode", "total", "tp", "fn", "fp", "tn", "precision", "recall", "f1"])
    for regime in data["regimes"]:
        writer.writerow([regime["key"], data["total"], *regime["matrix"].values(),
                         *(f"{regime[key]:.6f}" for key in ("precision", "recall", "f1"))])
    return output.getvalue()


def _svg(inner: str, height: int, title: str, *, dark: bool = False) -> str:
    background, border = ("#142826", "#476460") if dark else ("#f5efe2", "#c9c1af")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{height}" '
        f'viewBox="0 0 880 {height}" role="img" aria-labelledby="title">\n'
        f'<title id="title">{title}</title>\n'
        f'<rect x=".5" y=".5" width="879" height="{height-1}" rx="15" fill="{background}" stroke="{border}"/>\n'
        f'{inner}\n</svg>\n'
    )


def score_plate(data: dict, *, dark: bool = False) -> str:
    ink = "#f1ebd8" if dark else "#192a27"
    muted, grid = ("#bbcdc1", "#3f5954") if dark else ("#4c5c54", "#c9c1af")
    colours = {"tp": "#7caa9b" if dark else "#31705f", "fn": "#ef9a73" if dark else "#c35431",
               "fp": "#f0c377" if dark else "#926320", "tn": grid}
    rows = [f'<g font-family="system-ui, DejaVu Sans, sans-serif" fill="{ink}">',
            f'<text x="32" y="42" font-size="15" letter-spacing="2" fill="{muted}">03 / КОНТРОЛЬНАЯ ПЛАСТИНА</text>',
            '<text x="32" y="84" font-size="32" font-weight="700">Что нашли правила</text>',
            f'<text x="848" y="82" font-size="16" text-anchor="end" fill="{muted}">{data["total"]} ходов · CPU</text>',
            f'<path d="M32 101H848" stroke="{grid}"/>']
    for idx, mode in enumerate(data["regimes"]):
        y = 114 + idx * 56
        if mode["key"] == "cpu":
            backdrop = "#203b35" if dark else "#e0eadd"
            rows.append(f'<rect x="21" y="{y}" width="838" height="51" rx="6" fill="{backdrop}"/>')
        rows.append(f'<text x="32" y="{y+32}" font-size="19" font-weight="600">{mode["title"]}</text>')
        for cell, kind in enumerate(kind for kind, n in mode["matrix"].items() for _ in range(n)):
            x = 244 + cell * 11
            rows.append(f'<rect x="{x}" y="{y+17}" width="8" height="20" rx="1.5" fill="{colours[kind]}"/>')
        rows.append(f'<text x="847" y="{y+32}" text-anchor="end" font-size="19" font-weight="650">{mode["f1"]:.3f}</text>')
    rows.append(f'<text x="847" y="116" text-anchor="end" font-size="12" fill="{muted}">F1</text>')
    for i, (kind, label) in enumerate(zip(colours, ("Найдены", "Пропущены", "Ложные тревоги", "Верно чистые"))):
        x = 32 + 205 * i
        rows.append(f'<rect x="{x}" y="302" width="13" height="13" rx="2" fill="{colours[kind]}"/>')
        rows.append(f'<text x="{x+21}" y="314" font-size="15" fill="{muted}">{label}</text>')
    rows.append('</g>')
    summary = ", ".join(f'{mode["title"]}: F1 {mode["f1"]:.3f}' for mode in data["regimes"])
    return _svg("\n".join(rows), 344, f'{data["total"]} ходов, CPU. {summary}.', dark=dark)


def grouped(value: int) -> str:
    return f"{value:,}".replace(",", "\u2009")


def pack_plate(scene: dict) -> str:
    cap = scene["caps"][0]
    fonts = 'font-family="system-ui, DejaVu Sans, sans-serif"'
    ink, muted = "#192a27", "#4c5c54"
    rows = [
        '<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse">'
        '<path d="M0 5L5 0" stroke="#a47a4f" stroke-width="1.5"/></pattern></defs>',
        f'<g {fonts} fill="{ink}">',
        f'<text x="32" y="42" font-size="15" letter-spacing="2" fill="{muted}">02 / РЕНТГЕН КОНТЕКСТА</text>',
        '<text x="32" y="86" font-size="32" font-weight="700">Не просто отрезать хвост</text>',
        '<path d="M32 103H848" stroke="#c9c1af"/>',
        f'<text x="32" y="144" font-size="19" font-weight="600">До упаковки</text>',
        f'<text x="32" y="169" font-size="17" fill="{muted}">{grouped(scene["raw_chars"])} симв.</text>',
        f'<text x="32" y="235" font-size="19" font-weight="600">После упаковки</text>',
        f'<text x="32" y="261" font-size="17" fill="{muted}">{grouped(cap["chars"])} / {grouped(cap["budget"])} симв.</text>',
    ]
    for entry, result in zip(scene["entries"], cap["rows"]):
        x = 233 + (entry["number"]-1) * 44
        highlighted = entry["proof"]
        fill = {"whole": "#438071", "short": "url(#hatch)", "marker": "#f5efe2", "gone": "#f5efe2"}[result["state"]]
        colour = "#c35431" if highlighted else "#78988f"
        dash = ' stroke-dasharray="3 3"' if result["state"] in {"marker", "gone"} else ""
        rows.extend([
            f'<text x="{x+17}" y="132" font-size="13" text-anchor="middle" fill="{muted}">{entry["number"]:02}</text>',
            f'<rect x="{x}" y="144" width="34" height="28" rx="3" fill="{colour}"/>',
            f'<path d="M{x+17} 180v20" stroke="#b9b39f"/>',
            f'<rect x="{x}" y="214" width="34" height="28" rx="3" fill="{fill}" stroke="{colour}"{dash}/>',
        ])
        if result["state"] != "gone":
            rows.append(f'<circle cx="{x+17}" cy="228" r="{4 if highlighted else 2}" fill="{colour}"/>')
    rows.extend([
        '<path d="M32 278H848" stroke="#c9c1af"/>',
        f'<text x="32" y="310" font-size="17" fill="{muted}">Кадр {scene["proof_index"]:02} · контрольная точка {CHECKPOINT} '
        f'{"осталась в контексте" if cap["proof_present"] else "не сохранилась"}</text>',
        f'<text x="848" y="310" font-size="16" text-anchor="end" fill="{muted}">{cap["payloads"]} из {len(scene["entries"])} с содержимым</text>',
        '</g>',
    ])
    return _svg("\n".join(rows), 334, "Синтетический журнал до и после упаковки; оранжевый кадр — источник контрольной точки.")


def freeze_lab(data: dict) -> str:
    source = HERE / "src"
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    image = base64.b64encode((ASSETS / "comparator.jpg").read_bytes()).decode("ascii")
    template = (source / "lab.html").read_text(encoding="utf-8")
    substitutions = {
        "@@STYLE@@": (source / "lab.css").read_text(encoding="utf-8"),
        "@@SCRIPT@@": (source / "lab.js").read_text(encoding="utf-8"),
        "@@DATA@@": payload,
        "@@IMAGERY@@": image,
    }
    for marker, text in substitutions.items():
        if template.count(marker) != 1:
            raise ValueError(f"Expected one occurrence of {marker}")
        template = template.replace(marker, text)
    return template


def outputs(data: dict) -> dict[Path, bytes]:
    html = freeze_lab(data).encode("utf-8")
    archive = io.BytesIO()
    with ZipFile(archive, "w") as zip_file:
        member = ZipInfo("index.html")  # constant timestamp; reproducible download
        member.compress_type = ZIP_DEFLATED
        member.external_attr = 0o100644 << 16
        zip_file.writestr(member, html)
    return {
        HERE / "index.html": html,
        ASSETS / "guardian-lab.zip": archive.getvalue(),
        ASSETS / "scorecard.csv": score_csv(data["scores"]).encode("utf-8"),
        ASSETS / "score-sheet.svg": score_plate(data["scores"]).encode("utf-8"),
        ASSETS / "score-sheet-dark.svg": score_plate(data["scores"], dark=True).encode("utf-8"),
        ASSETS / "context-xray.svg": pack_plate(data["packing"]).encode("utf-8"),
    }


def collect() -> dict:
    config = RuntimeConfig.load(str(ROOT / "configs/default.json"))
    detector = Detector(config, backend_name="heuristic")
    return {"cases": export_cases(detector), "packing": pack_scene(config), "scores": scores(detector)}


def export_matches(path: Path, expected: bytes) -> bool:
    if not path.is_file():
        return False
    if path.suffix != ".zip":
        return path.read_bytes() == expected
    # Different zlib versions can produce different encodings of the same HTML.
    try:
        with ZipFile(path) as actual, ZipFile(io.BytesIO(expected)) as reference:
            return actual.namelist() == ["index.html"] and actual.read("index.html") == reference.read("index.html")
    except BadZipFile:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--check", action="store_true", help="verify the committed lab, CSV and SVGs")
    choices.add_argument("--art", action="store_true", help="also render cover, replay and print sheet (Pillow)")
    args = parser.parse_args()
    data = collect()
    stale = []
    for path, body in outputs(data).items():
        if args.check:
            if not export_matches(path, body):
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_bytes(body)
    if args.art:
        from art import export_art
        export_art(ASSETS, data)
    if stale:
        print("Stale exports (run python docs/showcase/build.py): " + ", ".join(stale), file=sys.stderr)
        return 1
    print("Field archive checked." if args.check else "Field archive exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
