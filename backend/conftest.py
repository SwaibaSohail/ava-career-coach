import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import config
import metering


@pytest.fixture(autouse=True)
def _temp_usage_ledger(tmp_path, monkeypatch):
    """Every test gets its own throwaway usage ledger; nothing touches data/usage.db.

    Client keys and the admin report are pinned to their defaults, so no test
    depends on what backend/.env happens to say."""
    monkeypatch.setattr(config, "METERING_DB", str(tmp_path / "usage.db"))
    monkeypatch.setattr(config, "REQUIRE_CLIENT_KEY", False)
    monkeypatch.setattr(config, "ADMIN_API_KEY", "")
    yield
    # Rows still queued for the writer thread belong in this test's ledger.
    metering.flush()


class FakeGroqCompletions:
    """Stands in for groq's chat.completions resource: canned replies, no network.

    content may be a list of pieces, streamed as one chunk each."""

    def __init__(self, content="ok", prompt_tokens=12, completion_tokens=5):
        self.pieces = content if isinstance(content, list) else [content]
        self.content = "".join(self.pieces)
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens
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
        first, *rest = self.pieces
        return [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": first},
                                  "finish_reason": None}]},
            *({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
              for piece in rest),
            # Groq reports usage on the final chunk, under x_groq.
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "x_groq": {"usage": self._usage()}},
        ]

    def create(self, messages, **params):
        self.calls.append(params)
        return iter(self._chunks(params)) if params.get("stream") else self._response(params)


class FakeAsyncGroqCompletions(FakeGroqCompletions):
    # Optional coroutine function awaited just before the usage chunk: a slow
    # reply (sleep), one that never finishes (wait forever) or a failure (raise).
    before_usage = None

    async def create(self, messages, **params):
        self.calls.append(params)
        if not params.get("stream"):
            return self._response(params)
        *chunks, usage = self._chunks(params)
        before_usage = self.before_usage

        async def gen():
            for chunk in chunks:
                yield chunk
            if before_usage is not None:
                await before_usage()
            yield usage
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
