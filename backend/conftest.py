import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
import pytest

import clef
import config
import guardrails
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


@pytest.fixture(autouse=True)
def _no_live_clef(monkeypatch):
    """No test reaches Cloudflare: a Clef call fails the test unless clef_api
    answers it. One made in the background, where the error may be caught (a
    shadow guard check), fails the test at teardown. Credentials are test
    values and the daily-quota pause is off."""
    background = []

    def refuse(request):
        if threading.current_thread() is not threading.main_thread():
            background.append(threading.current_thread().name)
        raise AssertionError("live Clef call; use the clef_api fixture")

    monkeypatch.setattr(clef, "_transport", httpx.MockTransport(refuse))
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "acct-test")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "tok-test")
    monkeypatch.setattr(clef, "_quota_until", None)
    yield
    assert not background, f"live Clef call from {background}; use the clef_api fixture"


@pytest.fixture(autouse=True)
def _guard_on_groq(tmp_path, monkeypatch):
    """The ingress guard runs as shipped (Groq, Clef settings at their defaults,
    no shadow text) whatever backend/.env says, with a throwaway decision log.
    Shadow checks a test started finish before the next test begins."""
    for name, value in {
        "GUARD_BACKEND": "groq", "CLEF_GUARD_MODEL": "clef", "CLEF_GUARD_RULE": "choice",
        "CLEF_BLOCK_THRESHOLD": 0.6, "CLEF_WINDOW_CHARS": 6000, "CLEF_FALLBACK": "groq",
        "CLEF_FALLBACK_TIMEOUT_S": 5.0, "GUARD_SHADOW_STORE_TEXT": False, "GUARD_SHADOW_RETENTION_DAYS": 14,
        "GUARD_LOG_DB": str(tmp_path / "guard.db"),
    }.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(guardrails, "_warned_no_clef", False)
    monkeypatch.setattr(guardrails, "_last_error_at", {})
    yield
    guardrails._shadow_drain(10)


class FakeClef:
    """Stands in for Cloudflare's Workers AI REST API: answers Clef calls as told,
    records each request (url, headers, json body, timeout), no network.

    Replies have the live-verified shape: {"result": {"model", "answers",
    "usage"}, "success", "errors", "messages"}."""

    def __init__(self):
        self.requests = []
        self.before = None   # optional callable(body), run before answering (a slow Clef)

    # Answers shaped like the live API's.
    @staticmethod
    def noul(p):
        return {"type": "noul", "noul": p}

    @staticmethod
    def choice(probabilities):
        return {"type": "choice", "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities, "confidence": 0.8}

    @staticmethod
    def score(probabilities, legend):
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probabilities)),
                "legend": {str(i): text for i, text in enumerate(legend)},
                "probabilities": {str(i): p for i, p in enumerate(probabilities)}, "confidence": 0.5}

    def reply(self, answers, input_tokens=346, envelope=True, request_id="req-test"):
        """Succeed with these answers (or answers(body), when callable)."""
        def respond(body):
            result = {"model": body["model"], "answers": answers(body) if callable(answers) else answers,
                      "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
            payload = {"result": result, "success": True, "errors": [], "messages": []} if envelope else result
            return httpx.Response(200, json=payload, headers={"cf-ai-req-id": request_id} if request_id else {})
        self._respond = respond

    def error(self, status, code=None, message="error"):
        """Fail as Cloudflare does: a status and a v4 envelope with one error."""
        payload = {"result": None, "success": False, "errors": [{"code": code, "message": message}],
                   "messages": []}
        self._respond = lambda body: httpx.Response(status, json=payload, headers={"cf-ai-req-id": "req-test"})

    def raw(self, status, text):
        self._respond = lambda body: httpx.Response(status, text=text)

    def raise_(self, exc):
        def respond(body):
            raise exc
        self._respond = respond

    def _respond(self, body):
        raise AssertionError("clef_api: set a reply first (reply, error, raw or raise_)")

    def handle(self, request):
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "headers": request.headers, "json": body,
                              "timeout": request.extensions.get("timeout")})
        if self.before is not None:
            self.before(body)
        return self._respond(body)


@pytest.fixture
def clef_api(monkeypatch):
    """A fake Workers AI endpoint for clef.py (see FakeClef)."""
    fake = FakeClef()
    monkeypatch.setattr(clef, "_transport", httpx.MockTransport(fake.handle))
    return fake


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
