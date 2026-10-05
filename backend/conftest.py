import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import config


@pytest.fixture(autouse=True)
def _temp_usage_ledger(tmp_path, monkeypatch):
    """Every test gets its own throwaway usage ledger; nothing touches data/usage.db."""
    monkeypatch.setattr(config, "METERING_DB", str(tmp_path / "usage.db"))


class FakeGroqCompletions:
    """Stands in for groq's chat.completions resource: canned replies, no network."""

    def __init__(self, content="ok", prompt_tokens=12, completion_tokens=5):
        self.content, self.prompt_tokens, self.completion_tokens = content, prompt_tokens, completion_tokens
        self.calls = []

    def _usage(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}

    def _response(self, params):
        return {"id": "fake", "object": "chat.completion", "created": 0, "model": params.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.content},
                             "finish_reason": "stop"}],
                "usage": self._usage()}

    def _chunks(self, params):
        base = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": params.get("model")}
        return [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": self.content},
                                  "finish_reason": None}]},
            # Groq reports usage on the final chunk, under x_groq.
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "x_groq": {"usage": self._usage()}},
        ]

    def create(self, messages, **params):
        self.calls.append(params)
        return iter(self._chunks(params)) if params.get("stream") else self._response(params)


class FakeAsyncGroqCompletions(FakeGroqCompletions):
    async def create(self, messages, **params):
        self.calls.append(params)
        if not params.get("stream"):
            return self._response(params)
        chunks = self._chunks(params)

        async def gen():
            for chunk in chunks:
                yield chunk
        return gen()


@pytest.fixture
def groq_llm(monkeypatch):
    """Factory for a real ChatGroq from llm.get_llm whose HTTP client is faked."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    import llm

    real_get_llm = llm.get_llm

    def make(content="ok", prompt_tokens=12, completion_tokens=5, **kwargs):
        model = real_get_llm(**kwargs)
        model.client = FakeGroqCompletions(content, prompt_tokens, completion_tokens)
        model.async_client = FakeAsyncGroqCompletions(content, prompt_tokens, completion_tokens)
        return model
    return make
