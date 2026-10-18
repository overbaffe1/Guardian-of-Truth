"""End-to-end prediction pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .backends import BackendError, BaseBackend, create_backend
from .config import RuntimeConfig
from .evidence import EvidenceConfig, build_evidence
from .fusion import FusionConfig, fuse
from .judge import JudgeItem, JudgeRunner
from .prompts import build_messages, build_verify_messages
from .signals import SignalsConfig, compute_signals, has_hard_signal
from .toolspec import extract_available_tools
from .transcript import parse_response, parse_transcript
from .types import Prediction, ResponseTurn, Signal, Transcript

logger = logging.getLogger(__name__)


@dataclass
class PreparedItem:
    uid: str  # internal unique key (row index based)
    transcript: Transcript
    response: ResponseTurn
    signals: list[Signal]
    external_id: Any = ""
    messages: Optional[list[dict[str, str]]] = None
    verify_messages: Optional[list[dict[str, str]]] = None
    evidence_chars: int = 0
    rebuild: Optional[Any] = None


class Detector:
    """Black-box response-level hallucination detector."""

    def __init__(
        self,
        config: RuntimeConfig,
        backend: Optional[BaseBackend] = None,
        auto_load_backend: bool = True,
        backend_name: str = "auto",
    ):
        self.config = config
        self.evidence_cfg: EvidenceConfig = config.evidence_config()
        self.signals_cfg: SignalsConfig = config.signals_config()
        self.fusion_cfg = FusionConfig.from_dict(
            {**config.fusion, "min_confidence": config.judge.get("min_confidence", 0.0)}
        )
        self.judge_cfg = dict(config.judge)
        self.judge_cfg["batch_size"] = config.model.get("batch_size", 16)
        self._backend = backend
        self._backend_name = getattr(backend, "name", "heuristic") if backend else "heuristic"
        self._backend_resolved = backend is not None or not auto_load_backend
        self._backend_requested = backend_name
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ backend
    def ensure_backend(self) -> Optional[BaseBackend]:
        if self._backend_resolved:
            return self._backend
        self._backend_resolved = True
        requested = self._backend_requested or self.config.raw.get("backend", "auto")
        try:
            self._backend = create_backend(str(requested), self.config.model)
            self._backend_name = self._backend.name if self._backend else "heuristic"
        except BackendError as exc:
            logger.error("No LLM backend available, falling back to rules: %s", exc)
            self._backend = None
            self._backend_name = "heuristic"
        return self._backend

    # ------------------------------------------------------------------ prepare
    def prepare(self, uid: str, prompt: str, response_text: str) -> PreparedItem:
        transcript = parse_transcript(prompt or "")
        response = parse_response(response_text or "")
        specs = extract_available_tools(transcript.system_text)
        signals = compute_signals(transcript, response, specs, self.signals_cfg)
        item = PreparedItem(uid=uid, transcript=transcript, response=response, signals=signals)
        if has_hard_signal(signals) and self.fusion_cfg.rule_mode == "strict":
            return item
        if not response.text.strip() and not response.tool_calls:
            return item
        evidence = build_evidence(transcript, response, specs, self.evidence_cfg)
        include_hints = bool(self.config.judge.get("include_hints", True))
        item.messages = build_messages(
            evidence.text,
            response,
            signals,
            include_hints=include_hints,
        )
        if bool(self.config.judge.get("verify_negatives", True)):
            item.verify_messages = build_verify_messages(
                evidence.text,
                response,
                signals,
                include_hints=include_hints,
            )
        item.evidence_chars = len(evidence.text)

        # Factory used by the runner to shrink the context if the run is
        # projected to exceed the 30-minute limit.

        def rebuild(scale: float, _t=transcript, _r=response, _s=specs, _sig=signals):
            import dataclasses

            smaller = dataclasses.replace(
                self.evidence_cfg,
                max_chars=max(4000, int(self.evidence_cfg.max_chars * scale)),
                max_tool_response_chars=max(
                    800, int(self.evidence_cfg.max_tool_response_chars * scale)
                ),
            )
            ev = build_evidence(_t, _r, _s, smaller)
            item.evidence_chars = len(ev.text)
            if item.verify_messages is not None:
                item.verify_messages = build_verify_messages(
                    ev.text, _r, _sig, include_hints=include_hints
                )
            return build_messages(ev.text, _r, _sig, include_hints=include_hints)

        item.rebuild = rebuild
        return item

    # ------------------------------------------------------------------ predict
    def predict_rows(
        self,
        rows: Iterable[dict[str, Any]],
        deadline: Optional[float] = None,
    ) -> list[Prediction]:
        started = time.monotonic()
        rows = list(rows)
        prepared: list[PreparedItem] = []
        errors = 0
        for index, row in enumerate(rows):
            external_id = row.get("id", index)
            uid = f"{index}:{external_id}"
            prompt_text = row.get("prompt")
            response_raw = row.get("response")
            if not isinstance(prompt_text, str):
                prompt_text = "" if prompt_text is None else str(prompt_text)
            if not isinstance(response_raw, str):
                response_raw = "" if response_raw is None else str(response_raw)
            try:
                item = self.prepare(uid, prompt_text, response_raw)
                item.external_id = external_id
                prepared.append(item)
            except Exception as exc:  # never fail the whole run on one row
                errors += 1
                logger.exception("Failed to prepare row %s: %s", uid, exc)
                prepared.append(
                    PreparedItem(
                        uid=uid,
                        external_id=external_id,
                        transcript=parse_transcript(""),
                        response=parse_response(""),
                        signals=[],
                    )
                )

        to_judge = [item for item in prepared if item.messages is not None]
        hard_items = [item for item in prepared if item.messages is None]
        logger.info(
            "Prepared %d rows: %d resolved by rules, %d sent to the judge, %d errors",
            len(prepared), len(hard_items), len(to_judge), errors,
        )

        self.ensure_backend()
        runner = JudgeRunner(self._backend, self.judge_cfg)

        budget = float(self.config.runtime.get("time_budget_sec", 1500))
        hard_deadline = deadline or (started + budget)
        # Keep a reserve for writing the output and for an emergency heuristic pass.
        judge_deadline = hard_deadline - min(90.0, budget * 0.08)

        judge_items = [
            JudgeItem(
                uid=item.uid,
                messages=item.messages or [],
                signals=item.signals,
                verify_messages=item.verify_messages,
                rebuild=item.rebuild,
            )
            for item in to_judge
        ]
        results = runner.run(judge_items, deadline=judge_deadline)

        predictions: list[Prediction] = []
        for item in prepared:
            if item.messages is None:
                prediction = fuse(
                    item.external_id, item.signals, None, self.fusion_cfg, self._backend_name
                )
            else:
                judge = results.get(item.uid)
                prediction = fuse(
                    item.external_id, item.signals, judge, self.fusion_cfg, self._backend_name
                )
            predictions.append(prediction)

        self.stats = {
            "rows": len(prepared),
            "rule_resolved": len(hard_items),
            "judged": len(to_judge),
            "errors": errors,
            "backend": self._backend_name,
            "failed_batches": runner.failed_batches,
            "verified": runner.verified,
            "verify_overrides": runner.verify_overrides,
            "elapsed_sec": round(time.monotonic() - started, 2),
        }
        return predictions

    # ------------------------------------------------------------------ debug
    def explain(self, prompt: str, response_text: str) -> dict[str, Any]:
        item = self.prepare("debug", prompt, response_text)
        return {
            "signals": [
                {"kind": s.kind, "severity": s.severity, "message": s.message, "evidence": s.evidence}
                for s in item.signals
            ],
            "domain": item.transcript.domain,
            "evidence_chars": item.evidence_chars,
            "messages": item.messages,
        }
