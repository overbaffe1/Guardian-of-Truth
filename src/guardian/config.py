"""Configuration loading and merging."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .evidence import EvidenceConfig
from .signals import SignalsConfig

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "path": "",  # optional local directory with weights
        "max_model_len": 32768,
        "max_prompt_tokens": 18000,
        "gpu_memory_utilization": 0.90,
        "dtype": "bfloat16",
        "load_in_4bit": False,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "batch_size": 32,
    },
    "evidence": EvidenceConfig().__dict__,
    "signals": SignalsConfig().__dict__,
    "judge": {
        "temperature": 0.0,
        "max_tokens": 448,
        "self_consistency": 1,       # 1 = greedy single pass
        "consistency_temperature": 0.4,
        "min_confidence": 0.0,        # below this the judge verdict is ignored
        "retry_on_parse_error": True,
        "include_hints": True,
        # Second-opinion pass: responses cleared by the first pass are re-checked
        # with a claim-by-claim prompt (recovers unsupported claims cheaply).
        "verify_negatives": True,
        "verify_threshold": 0.65,
        "verify_max_tokens": 384,
        "verify_temperature": 0.0,
    },
    "fusion": {
        "threshold": 0.5,
        "rule_mode": "strict",       # strict: hard signals decide; hint: only hints
        "soft_signal_override": 0.0,  # >0: sum of soft weights needed to override judge
    },
    "runtime": {
        "time_budget_sec": 1500,
        "log_every": 25,
        "fallback_on_backend_error": True,
    },
    "backend": "auto",  # auto | vllm | transformers | heuristic
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class RuntimeConfig:
    raw: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[dict] = None) -> "RuntimeConfig":
        data = copy.deepcopy(DEFAULT_CONFIG)
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = _deep_merge(data, json.load(fh))
        # Environment variables are handy inside Docker.
        env_model = os.environ.get("GUARDIAN_MODEL_ID")
        if env_model:
            data["model"]["id"] = env_model
        env_path = os.environ.get("GUARDIAN_MODEL_PATH")
        if env_path:
            data["model"]["path"] = env_path
        if overrides:
            data = _deep_merge(data, {k: v for k, v in overrides.items() if v is not None})
        return cls(raw=data)

    # convenience accessors -------------------------------------------------
    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def judge(self) -> dict[str, Any]:
        return self.raw["judge"]

    @property
    def fusion(self) -> dict[str, Any]:
        return self.raw["fusion"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]

    def evidence_config(self) -> EvidenceConfig:
        cfg = EvidenceConfig.from_dict(self.raw.get("evidence"))
        max_tokens = int(self.model.get("max_prompt_tokens", 12000))
        # Config stores chars; keep both knobs consistent.
        cfg.max_chars = min(cfg.max_chars, int(max_tokens * 3.0))
        return cfg

    def signals_config(self) -> SignalsConfig:
        return SignalsConfig.from_dict(self.raw.get("signals"))

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.raw, fh, ensure_ascii=False, indent=2)
