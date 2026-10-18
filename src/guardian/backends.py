"""Inference backends (vLLM, transformers, or none).

Imports of torch / vllm / transformers are lazy so that the rule-only path
and the unit tests never require a GPU stack.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from .prompts import fit_user_content

logger = logging.getLogger(__name__)


class BackendError(RuntimeError):
    pass


@dataclass
class GenerationRequest:
    messages: list[dict[str, str]]
    max_tokens: int = 640
    temperature: float = 0.0
    n: int = 1


class BaseBackend:
    name = "base"

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional cleanup
        pass


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


class VLLMBackend(BaseBackend):
    name = "vllm"

    def __init__(self, model_path: str, cfg: dict[str, Any]):
        try:
            from vllm import LLM  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise BackendError(f"vllm is not installed: {exc}") from exc

        self.max_model_len = int(cfg.get("max_model_len", 32768))
        kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
            "dtype": cfg.get("dtype", "bfloat16"),
            "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.9)),
            "max_model_len": self.max_model_len,
            # The system block and the policy are shared by all examples of a
            # domain; prefix caching removes most of the prefill cost.
            "enable_prefix_caching": bool(cfg.get("enable_prefix_caching", True)),
        }
        # AWQ/GPTQ checkpoints are detected automatically by vLLM; only pass
        # ``quantization`` when the user explicitly configured it.
        if cfg.get("quantization"):
            kwargs["quantization"] = cfg["quantization"]
        logger.info("Loading vLLM model %s with %s", model_path, kwargs)
        try:
            self._llm = LLM(**kwargs)
        except TypeError as exc:
            # Older vLLM builds do not know ``enable_prefix_caching``.
            if "enable_prefix_caching" in str(exc):
                kwargs.pop("enable_prefix_caching", None)
                logger.warning("Prefix caching is not supported by this vLLM build")
                self._llm = LLM(**kwargs)
            else:
                raise
        self._tokenizer = self._llm.get_tokenizer()
        self._max_prompt_tokens = int(cfg.get("max_prompt_tokens", 12000))

    def _render(self, messages: list[dict[str, str]]) -> str:
        tokenizer = self._tokenizer
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"
        # Hard length guard: character-level truncation of the user message.
        try:
            ids = tokenizer.encode(prompt)
        except Exception:
            ids = []
        limit = min(self._max_prompt_tokens, self.max_model_len - 1024)
        if ids and len(ids) > limit:
            ratio = len(prompt) / max(1, len(ids))
            max_chars = int(limit * ratio * 0.95)
            messages = list(messages)
            messages[-1] = {
                "role": messages[-1]["role"],
                "content": fit_user_content(messages[-1]["content"], max_chars),
            }
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        from vllm import SamplingParams

        prompts = [self._render(r.messages) for r in requests]
        # Group by sampling parameters (vLLM requires one params per call).
        results: list[list[str]] = [[] for _ in requests]
        groups: dict[tuple[float, int], list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault((float(r.temperature), int(r.n)), []).append(i)

        for (temperature, n), indices in groups.items():
            params = SamplingParams(
                temperature=temperature,
                max_tokens=max(r.max_tokens for r in requests),
                n=n,
                top_p=1.0 if temperature == 0 else 0.95,
            )
            outputs = self._llm.generate(
                [prompts[i] for i in indices], params, use_tqdm=False
            )
            for idx, output in zip(indices, outputs):
                results[idx] = [o.text for o in output.outputs]
        return results


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


class TransformersBackend(BaseBackend):
    name = "transformers"

    def __init__(self, model_path: str, cfg: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise BackendError(f"transformers/torch are not available: {exc}") from exc

        self._torch = torch
        self.max_model_len = int(cfg.get("max_model_len", 32768))
        self.max_prompt_tokens = int(cfg.get("max_prompt_tokens", 12000))
        logger.info("Loading transformers model %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=bool(cfg.get("trust_remote_code", True))
        )
        self._tokenizer.padding_side = "left"
        # Keep the end of the prompt (response + hints) when truncating.
        try:
            self._tokenizer.truncation_side = "left"
        except Exception:
            pass
        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(cfg.get("trust_remote_code", True)),
            "device_map": "auto",
        }
        if cfg.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self._model.eval()

    def _render(self, messages: list[dict[str, str]]) -> str:
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"
        return prompt

    def generate(self, requests: list[GenerationRequest]) -> list[list[str]]:
        torch = self._torch
        results: list[list[str]] = []
        device = next(self._model.parameters()).device
        for request in requests:
            prompt = self._render(request.messages)
            ids = self._tokenizer(prompt)["input_ids"]
            if len(ids) > self.max_prompt_tokens:
                ratio = len(prompt) / max(1, len(ids))
                max_chars = int(self.max_prompt_tokens * ratio * 0.95)
                messages = list(request.messages)
                messages[-1] = {
                    "role": messages[-1]["role"],
                    "content": fit_user_content(messages[-1]["content"], max_chars),
                }
                prompt = self._render(messages)
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_prompt_tokens
            ).to(device)
            do_sample = request.temperature > 0
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    do_sample=do_sample,
                    temperature=request.temperature if do_sample else None,
                    num_return_sequences=request.n,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            texts = [
                self._tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                for seq in output
            ]
            results.append(texts)
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def resolve_model_path(cfg: dict[str, Any]) -> str:
    explicit = (cfg.get("path") or "").strip()
    if explicit:
        return explicit
    env = os.environ.get("GUARDIAN_MODEL_PATH", "").strip()
    if env:
        return env
    # default location used by the Docker image / download script
    local = os.path.join(os.getcwd(), "model")
    if os.path.isdir(local) and any(
        name.endswith((".safetensors", ".bin")) for name in os.listdir(local)
    ):
        return local
    return str(cfg.get("id"))


def create_backend(
    requested: str, cfg: dict[str, Any]
) -> Optional[BaseBackend]:
    """Create a backend or return ``None`` for the heuristic fallback."""
    requested = (requested or "auto").lower()
    if requested == "heuristic":
        return None
    model_path = resolve_model_path(cfg)

    errors: list[str] = []
    candidates = ["vllm", "transformers"] if requested == "auto" else [requested]
    for candidate in candidates:
        try:
            if candidate == "vllm":
                return VLLMBackend(model_path, cfg)
            if candidate == "transformers":
                return TransformersBackend(model_path, cfg)
            raise BackendError(f"unknown backend: {candidate}")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            logger.warning("Backend %s unavailable: %s", candidate, exc)
    raise BackendError("; ".join(errors) or "no backend available")


def wait_until(deadline: float, poll: float = 0.5) -> bool:
    """Return True when the deadline has passed."""
    return time.monotonic() >= deadline
