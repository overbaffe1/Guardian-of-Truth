"""Tests for the vLLM backend using an in-memory fake of the vLLM API."""

import sys
import types

import pytest

from guardian.backends import GenerationRequest, VLLMBackend
from guardian.prompts import fit_user_content


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    def encode(self, text):
        # ~3 characters per token
        return list(range(max(1, len(text) // 3)))


class _Output:
    def __init__(self, text, n=1):
        self.outputs = [types.SimpleNamespace(text=text) for _ in range(n)]


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        if kwargs.get("_fail_prefix_cache"):
            raise TypeError("unexpected keyword argument 'enable_prefix_caching'")
        if "enable_prefix_caching" in kwargs and FakeLLM.reject_prefix_cache:
            raise TypeError("unexpected keyword argument 'enable_prefix_caching'")
        self.kwargs = kwargs
        self.prompts = []
        FakeLLM.instances.append(self)

    reject_prefix_cache = False

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params, use_tqdm=False):
        self.prompts.extend(prompts)
        n = getattr(params, "kwargs", {}).get("n", 1)
        return [_Output('{"label": 0, "confidence": 0.9}', n) for _ in prompts]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def fake_vllm(monkeypatch):
    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    FakeLLM.instances.clear()
    FakeLLM.reject_prefix_cache = False
    yield module
    FakeLLM.instances.clear()


def test_vllm_backend_generates_and_groups(fake_vllm):
    backend = VLLMBackend("fake-model", {"max_model_len": 32768, "max_prompt_tokens": 12000})
    requests = [
        GenerationRequest(messages=[{"role": "user", "content": "a"}], n=1, temperature=0.0),
        GenerationRequest(messages=[{"role": "user", "content": "b"}], n=2, temperature=0.4),
        GenerationRequest(messages=[{"role": "user", "content": "c"}], n=1, temperature=0.0),
    ]
    outputs = backend.generate(requests)
    assert len(outputs) == 3
    assert outputs[1] == ['{"label": 0, "confidence": 0.9}'] * 2


def test_vllm_backend_keeps_response_when_truncating(fake_vllm):
    backend = VLLMBackend("fake-model", {"max_model_len": 32768, "max_prompt_tokens": 500})
    content = (
        "[CONTEXT]\n"
        + ("context line\n" * 5000)
        + "[RESPONSE UNDER AUDIT]\n<response_under_audit>\nKEEP_ME\n</response_under_audit>\n"
    )
    backend.generate([GenerationRequest(messages=[{"role": "user", "content": content}])])
    prompt = FakeLLM.instances[-1].prompts[-1]
    assert "KEEP_ME" in prompt
    assert len(prompt) < len(content)


def test_prefix_cache_fallback(fake_vllm):
    FakeLLM.reject_prefix_cache = True
    backend = VLLMBackend("fake-model", {"max_model_len": 32768})
    assert backend is not None
    assert "enable_prefix_caching" not in FakeLLM.instances[-1].kwargs


def test_fit_user_content_never_exceeds_by_much():
    content = "[CONTEXT]\n" + "x" * 10_000 + "\n[RESPONSE UNDER AUDIT]\nRESP\n"
    fitted = fit_user_content(content, 1000)
    assert "RESP" in fitted
    assert len(fitted) <= 1100
