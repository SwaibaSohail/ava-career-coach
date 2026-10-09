import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone

import anyio
import httpx
import pytest
from starlette.concurrency import run_in_threadpool

import clef
import config as app_config
import guard_log
import guardrails
import metering
from test_guardrails import _clef_answers, _decisions, _ledger, CLEAN_P, INJECTION_P


def _groq_http(monkeypatch, responder):
    monkeypatch.setattr(app_config, "GROQ_API_KEY", "test-key")
    sent = []

    def handle(transport, request):
        sent.append(threading.current_thread().name)
        return responder(request)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return sent


def _groq_says(label):
    return httpx.Response(200, json={
        "id": "fake", "object": "chat.completion", "created": 0, "model": app_config.GUARD_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": label}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}})


def _through_starlette(fn, *args):
    async def main():
        with metering.context(client_id="acme", session_id="s1", feature="chat"):
            return await run_in_threadpool(fn, *args)
    return anyio.run(main)


def _rows():
    return [(r["provider"], r["feature"], r["client_id"], r["session_id"]) for r in _ledger()]


S1 = metering.session_ref("s1")


def test_groq_mode_ledger(monkeypatch):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "groq")
    sent = _groq_http(monkeypatch, lambda r: _groq_says("CLEAN"))
    assert _through_starlette(guardrails.check_input_llm, "How do I improve my CV?") == "CLEAN"
    print("threads", sent)
    assert _rows() == [("groq", "guard", "acme", S1)]


def test_groq_mode_abandoned_attempt_ledger(monkeypatch):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "groq")
    monkeypatch.setattr(app_config, "GUARD_TIMEOUT_S", 0.3)
    monkeypatch.setattr(app_config, "GUARD_MAX_RETRIES", 1)
    first = threading.Event()
    n = [0]

    def responder(request):
        n[0] += 1
        if n[0] == 1:
            first.wait(3)
        return _groq_says("CLEAN")
    _groq_http(monkeypatch, responder)
    assert _through_starlette(guardrails.check_input_llm, "How do I improve my CV?") == "CLEAN"
    first.set()
    time.sleep(0.5)
    assert sorted(_rows()) == [("groq", "guard", "acme", S1)] * 2


def test_clef_mode_fallback_ledger(clef_api, monkeypatch):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "clef")
    clef_api.error(503)
    _groq_http(monkeypatch, lambda r: _groq_says("ABUSE"))
    assert _through_starlette(guardrails.check_input_llm, "How do I improve my CV?") == "ABUSE"
    assert _rows() == [("groq", "guard", "acme", S1)]


def test_clef_mode_ledger(clef_api, monkeypatch):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "clef")
    monkeypatch.setattr(app_config, "CLEF_WINDOW_CHARS", 10)
    clef_api.reply(_clef_answers(clef_api))
    assert _through_starlette(guardrails.check_input_llm, "How do I improve my CV?") == "CLEAN"
    assert _rows() == [("cloudflare", "guard", "acme", S1)] * 2


def test_shadow_mode_ledger(clef_api, monkeypatch):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "shadow")
    monkeypatch.setattr(app_config, "CLEF_WINDOW_CHARS", 10)
    clef_api.reply(_clef_answers(clef_api))
    _groq_http(monkeypatch, lambda r: _groq_says("CLEAN"))
    assert _through_starlette(guardrails.check_input_llm, "How do I improve my CV?") == "CLEAN"
    assert guardrails._shadow_drain(5)
    assert sorted(_rows()) == sorted([("groq", "guard", "acme", S1)] + [("cloudflare", "guard.shadow", "acme", S1)] * 2)


def test_shadow_groq_fails(clef_api, monkeypatch):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "shadow")
    monkeypatch.setattr(app_config, "GUARD_SHADOW_STORE_TEXT", True)
    monkeypatch.setattr(guardrails, "_RETRY_PAUSE_S", 0)
    clef_api.reply(_clef_answers(clef_api, INJECTION_P))
    _groq_http(monkeypatch, lambda r: httpx.Response(500, json={"error": {"message": "x"}}))
    assert _through_starlette(guardrails.check_input_llm, "How do I improve my CV?") == "CLEAN"
    assert guardrails._shadow_drain(5)
    (d,) = _decisions()
    print(d)
    assert d["groq_label"] is None


def test_shadow_unexpected_failure_row(monkeypatch, caplog):
    monkeypatch.setattr(app_config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "shadow")
    monkeypatch.setattr(guardrails, "_groq_label", lambda m, **k: "CLEAN")

    def broken(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(clef, "decide", broken)
    assert guardrails.check_input_llm("How do I improve my CV?") == "CLEAN"
    assert guardrails._shadow_drain(5)
    print("shadow decisions:", _decisions())
    # Same failure in clef mode:
    monkeypatch.setattr(app_config, "GUARD_BACKEND", "clef")
    monkeypatch.setattr(guardrails, "_groq_guard", lambda m, **k: "CLEAN")
    guardrails.check_input_llm("How do I improve my CV?")
    print("all decisions:", [(r["mode"], r["error"]) for r in _decisions()])


def test_eval_ask_groq_tokens(monkeypatch):
    from evals import guard_eval
    _groq_http(monkeypatch, lambda r: _groq_says("CLEAN"))
    with metering.feature("eval.guard"):
        ans = guard_eval._ask_groq("How do I improve my CV?")
    print("eval answer", ans)
    assert ans["input_tokens"] == 12
    assert _rows() == [("groq", "eval.guard", "default", None)]


def test_eval_ask_clef_ledger(clef_api, monkeypatch):
    from evals import guard_eval
    clef_api.reply(_clef_answers(clef_api))
    with metering.feature("eval.guard"):
        ans = guard_eval._ask_clef("How do I improve my CV?", "clef")
    assert _rows() == [("cloudflare", "eval.guard", "default", None)]


def test_quota_sim(clef_api, monkeypatch, caplog):
    from datetime import timedelta
    from test_clef import STATE, QUESTIONS
    now = [datetime(2026, 10, 8, 23, 59, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(clef, "_utcnow", lambda: now[0])
    clef_api.error(429, 3036)
    sent_at = []
    for i in range(20 * 60):
        before = len(clef_api.requests)
        try:
            clef.decide(STATE, QUESTIONS)
        except clef.ClefError:
            pass
        if len(clef_api.requests) > before:
            sent_at.append(now[0].strftime("%H:%M:%S"))
        now[0] += timedelta(seconds=1)
    print("sent at", sent_at)
    print("until", clef.quota_exhausted_until())
    print([r.levelname + ' ' + r.getMessage()[:60] for r in caplog.records if r.name == "clef"])
