import asyncio
import contextvars
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys

import pytest
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from pydantic import BaseModel

import config
import guardrails
import interview
import llm
import manage_clients
import metering


def _rows():
    conn = sqlite3.connect(config.METERING_DB)
    conn.row_factory = sqlite3.Row
    try:
        # Nothing recorded yet means no schema either: an empty ledger.
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'usage'").fetchone():
            return []
        return [dict(r) for r in conn.execute("SELECT * FROM usage ORDER BY id")]
    finally:
        conn.close()


def _record_at(monkeypatch, ts, model, inp, out, **ctx):
    monkeypatch.setattr(metering, "_now", lambda: ts)
    with metering.context(**ctx):
        metering.record("groq", model, inp, out)


def _record_for(client_id, tokens):
    with metering.context(client_id=client_id):
        metering.record("groq", "openai/gpt-oss-120b", tokens, 0)


# --- Attribution ----------------------------------------------------------

def test_record_uses_context_and_defaults():
    metering.record("groq", "openai/gpt-oss-120b", 100, 20)
    with metering.context(client_id="acme", session_id="s1", feature="chat"):
        metering.record("groq", "openai/gpt-oss-120b", 10, 5)
    first, second = _rows()
    assert (first["client_id"], first["session_id"], first["feature"]) == ("default", None, "unknown")
    assert (second["client_id"], second["session_id"], second["feature"]) == ("acme", "s1", "chat")
    assert second["total_tokens"] == 15 and second["provider"] == "groq"
    assert first["ts_utc"].endswith("Z")


def test_ledger_stores_no_message_text():
    with metering.context(client_id="acme", session_id="s1", feature="chat"):
        metering.record("groq", "openai/gpt-oss-120b", 10, 5)
    assert set(_rows()[0]) == {
        "id", "ts_utc", "client_id", "session_id", "feature", "provider", "model",
        "input_tokens", "output_tokens", "total_tokens", "cost_usd",
    }


def test_feature_overrides_only_feature_and_resets():
    with metering.context(client_id="acme", feature="chat"):
        with metering.feature("guard"):
            assert metering.current()["feature"] == "guard"
            assert metering.current()["client_id"] == "acme"
        assert metering.current()["feature"] == "chat"
    assert metering.current() == {"client_id": "default", "session_id": None, "feature": "unknown"}


def test_context_propagates_through_run_in_threadpool():
    async def go():
        with metering.context(client_id="acme", session_id="s9", feature="chat"):
            return await run_in_threadpool(metering.current)
    assert asyncio.run(go()) == {"client_id": "acme", "session_id": "s9", "feature": "chat"}


def test_context_propagates_into_asyncio_tasks():
    async def go():
        with metering.context(client_id="acme"):
            return await asyncio.create_task(asyncio.to_thread(metering.current))
    assert asyncio.run(go())["client_id"] == "acme"


def test_context_exit_in_another_context_does_not_raise():
    cm = metering.context(client_id="acme")
    contextvars.copy_context().run(cm.__enter__)
    contextvars.copy_context().run(cm.__exit__, None, None, None)  # must not raise


# --- Cost -----------------------------------------------------------------

def test_prices_and_plans_load_without_notes():
    assert "_note" not in metering.prices() and "_note" not in metering.plans()
    assert metering.plans()["internal"]["monthly_tokens"] is None
    assert metering.plans()["free"]["monthly_tokens"] == 100_000


def test_estimate_cost_for_known_model():
    assert metering.estimate_cost("openai/gpt-oss-120b", 1_000_000, 1_000_000) == 0.75


def test_unknown_model_cost_is_null_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(metering, "_warned_models", set())
    with caplog.at_level(logging.WARNING, logger="metering"):
        assert metering.estimate_cost("mystery-model", 100, 10) is None
        assert metering.estimate_cost("mystery-model", 200, 20) is None
    assert sum("no price" in r.getMessage() for r in caplog.records) == 1


def test_unknown_model_row_records_null_cost():
    metering.record("groq", "mystery-model", 100, 10)
    (row,) = _rows()
    assert row["cost_usd"] is None and row["total_tokens"] == 110


def test_malformed_price_entry_still_records_tokens(monkeypatch):
    monkeypatch.setattr(metering, "_prices", {"half-priced": {"input": 0.1}})
    monkeypatch.setattr(metering, "_warned_models", set())
    metering.record("groq", "half-priced", 100, 10)
    (row,) = _rows()
    assert row["cost_usd"] is None and row["total_tokens"] == 110


# --- Months ---------------------------------------------------------------

def test_month_bounds():
    assert metering.month_bounds("2026-12") == (
        "2026-12", "2026-12-01T00:00:00.000000Z", "2027-01-01T00:00:00.000000Z")
    assert metering.month_bounds("2026-02") == (
        "2026-02", "2026-02-01T00:00:00.000000Z", "2026-03-01T00:00:00.000000Z")
    label, start, end = metering.month_bounds()
    assert start < metering._now() < end and start.startswith(label)


@pytest.mark.parametrize("bad", ["2026-13", "Oct", "2026-1", "", "2026-10-01"])
def test_month_bounds_rejects_bad_months(bad):
    with pytest.raises(ValueError):
        metering.month_bounds(bad)


def test_month_boundary_is_utc(monkeypatch):
    _record_at(monkeypatch, "2026-10-31T23:59:59.999999Z", "openai/gpt-oss-120b", 100, 0)
    _record_at(monkeypatch, "2026-11-01T00:00:00.000000Z", "openai/gpt-oss-120b", 7, 0)
    october = metering.usage_report("2026-10")
    november = metering.usage_report("2026-11")
    assert [r["total_tokens"] for r in october["breakdown"]] == [100]
    assert [r["total_tokens"] for r in november["breakdown"]] == [7]


# --- Clients --------------------------------------------------------------

def test_add_client_and_look_up_by_key():
    client_id, key = metering.add_client("Acme Ltd", "starter")
    assert key.startswith("ava_") and client_id.startswith("acme-ltd-")
    assert metering.client_for_key(key) == client_id
    assert metering.client_for_key("nope") is None
    assert metering.client_for_key("") is None
    assert metering.client_for_key(None) is None

    conn = sqlite3.connect(config.METERING_DB)
    try:
        (stored,) = conn.execute("SELECT key_hash FROM clients WHERE client_id = ?", (client_id,)).fetchone()
    finally:
        conn.close()
    assert stored != key
    assert stored == hashlib.sha256(key.encode()).hexdigest()


def test_unknown_plans_and_clients_are_rejected():
    with pytest.raises(ValueError, match="unknown plan"):
        metering.add_client("Acme", "gold")
    client_id, _ = metering.add_client("Acme", "free")
    with pytest.raises(ValueError, match="unknown plan"):
        metering.set_plan(client_id, "gold")
    with pytest.raises(ValueError, match="unknown client"):
        metering.set_plan("nobody-123456", "pro")


def test_set_plan_and_list_clients():
    client_id, _ = metering.add_client("Acme", "free")
    metering.set_plan(client_id, "pro")
    plans = {c["client_id"]: c["plan"] for c in metering.list_clients()}
    assert plans == {"default": "internal", client_id: "pro"}
    assert all("key_hash" not in c for c in metering.list_clients())


def test_default_client_is_internal_and_never_capped():
    _record_for(metering.DEFAULT_CLIENT, 10_000_000)
    assert {c["client_id"]: c["plan"] for c in metering.list_clients()}["default"] == "internal"
    allowance = metering.check_allowance("default")
    assert allowance == metering.Allowance(allowed=True, used=10_000_000, limit=None)


# --- Allowances -----------------------------------------------------------

def test_free_plan_blocks_at_its_limit():
    client_id, _ = metering.add_client("Acme", "free")
    _record_for(client_id, 99_999)
    assert metering.check_allowance(client_id) == metering.Allowance(True, 99_999, 100_000)
    _record_for(client_id, 1)
    assert metering.check_allowance(client_id) == metering.Allowance(False, 100_000, 100_000)


def test_usage_from_other_clients_does_not_count():
    client_id, _ = metering.add_client("Acme", "free")
    _record_for("someone-else", 500_000)
    assert metering.check_allowance(client_id).allowed


def test_suspended_client_is_blocked_with_no_usage():
    client_id, _ = metering.add_client("Acme", "suspended")
    assert metering.check_allowance(client_id) == metering.Allowance(False, 0, 0)


def test_client_on_unknown_plan_is_allowed_with_warning(monkeypatch, caplog):
    client_id, _ = metering.add_client("Acme", "starter")
    monkeypatch.setattr(metering, "_plans", {"free": {"monthly_tokens": 100_000}})
    with caplog.at_level(logging.WARNING, logger="metering"):
        allowance = metering.check_allowance(client_id)
    assert allowance.allowed and allowance.limit is None
    assert "unknown plan" in caplog.text


def test_allowance_check_fails_open(monkeypatch, caplog):
    def boom():
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(metering, "_connect", boom)
    with caplog.at_level(logging.ERROR, logger="metering"):
        assert metering.check_allowance("anyone").allowed is True
    assert "allowance check failed" in caplog.text


# --- Report ---------------------------------------------------------------

def test_usage_report_groups_by_client_feature_model_and_day(monkeypatch):
    acme, _ = metering.add_client("Acme", "starter")
    beta, _ = metering.add_client("Beta", "free")
    big = "openai/gpt-oss-120b"
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", big, 1000, 200, client_id=acme, feature="chat")
    _record_at(monkeypatch, "2026-10-06T18:00:00.000000Z", big, 500, 100, client_id=acme, feature="chat")
    _record_at(monkeypatch, "2026-10-07T08:00:00.000000Z", big, 10, 1, client_id=acme, feature="chat")
    _record_at(monkeypatch, "2026-10-06T09:00:01.000000Z", "llama-3.1-8b-instant", 300, 1,
               client_id=acme, feature="guard")
    _record_at(monkeypatch, "2026-10-06T10:00:00.000000Z", "mystery-model", 50, 5, client_id=beta, feature="chat")

    report = metering.usage_report("2026-10")
    assert report["month"] == "2026-10" and report["timezone"] == "UTC"
    assert report["period_start_utc"] == "2026-10-01T00:00:00.000000Z"
    assert report["period_end_utc"] == "2026-11-01T00:00:00.000000Z"
    assert "exact" in report["note"] and "estimate" in report["note"]

    groups = {(r["client_id"], r["feature"], r["model"], r["day"]): r for r in report["breakdown"]}
    assert set(groups) == {
        (acme, "chat", big, "2026-10-06"),
        (acme, "chat", big, "2026-10-07"),
        (acme, "guard", "llama-3.1-8b-instant", "2026-10-06"),
        (beta, "chat", "mystery-model", "2026-10-06"),
    }
    chat = groups[(acme, "chat", big, "2026-10-06")]
    assert (chat["calls"], chat["input_tokens"], chat["output_tokens"], chat["total_tokens"]) == (2, 1500, 300, 1800)
    assert chat["cost_usd"] == pytest.approx(metering.estimate_cost(big, 1500, 300))
    assert chat["unpriced_calls"] == 0
    unpriced = groups[(beta, "chat", "mystery-model", "2026-10-06")]
    assert unpriced["cost_usd"] is None and unpriced["unpriced_calls"] == 1

    clients = {c["client_id"]: c for c in report["clients"]}
    assert set(clients) == {"default", acme, beta}
    assert clients[acme]["name"] == "Acme" and clients[acme]["plan"] == "starter"
    assert clients[acme]["used_tokens"] == 1800 + 11 + 301
    assert clients[acme]["input_tokens"] == 1810 and clients[acme]["output_tokens"] == 302
    assert clients[acme]["monthly_tokens"] == 1_000_000
    assert clients[acme]["remaining_tokens"] == 1_000_000 - 2112
    assert clients[acme]["cost_usd"] == pytest.approx(
        metering.estimate_cost(big, 1510, 301) + metering.estimate_cost("llama-3.1-8b-instant", 300, 1))
    assert clients[beta]["cost_usd"] is None and clients[beta]["unpriced_calls"] == 1
    assert clients[beta]["remaining_tokens"] == 100_000 - 55
    assert clients["default"]["used_tokens"] == 0 and clients["default"]["remaining_tokens"] is None


def test_usage_report_remaining_never_goes_negative(monkeypatch):
    client_id, _ = metering.add_client("Acme", "free")
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 150_000, 0, client_id=client_id)
    (acme,) = [c for c in metering.usage_report("2026-10")["clients"] if c["client_id"] == client_id]
    assert acme["used_tokens"] == 150_000 and acme["remaining_tokens"] == 0


def test_usage_report_filters_by_client(monkeypatch):
    acme, _ = metering.add_client("Acme", "starter")
    beta, _ = metering.add_client("Beta", "free")
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 10, 1, client_id=acme)
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 20, 2, client_id=beta)
    report = metering.usage_report("2026-10", client_id=beta)
    assert [c["client_id"] for c in report["clients"]] == [beta]
    assert [r["client_id"] for r in report["breakdown"]] == [beta]


def test_usage_report_includes_usage_from_unregistered_clients(monkeypatch):
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 40, 2, client_id="ghost")
    clients = {c["client_id"]: c for c in metering.usage_report("2026-10")["clients"]}
    assert clients["ghost"]["used_tokens"] == 42
    assert clients["ghost"]["plan"] is None and clients["ghost"]["remaining_tokens"] is None


def test_usage_report_rejects_bad_month():
    with pytest.raises(ValueError):
        metering.usage_report("2026-13")


# --- Recorder -------------------------------------------------------------

def _streamed_result(inp=40, out=9):
    # Shape ChatGroq produces for a streamed call: aggregated chunk with usage, no llm_output.
    msg = AIMessageChunk(content="hi", usage_metadata={"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out})
    return LLMResult(generations=[[ChatGenerationChunk(message=msg)]], llm_output=None)


def test_recorder_records_streamed_usage():
    with metering.context(client_id="acme", feature="chat"):
        metering.UsageRecorder("groq", "openai/gpt-oss-120b").on_llm_end(_streamed_result())
    (row,) = _rows()
    assert row["input_tokens"] == 40 and row["output_tokens"] == 9 and row["client_id"] == "acme"


def test_recorder_falls_back_to_llm_output_token_usage():
    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]],
                       llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    metering.UsageRecorder("groq", "m").on_llm_end(result)
    assert (_rows()[0]["input_tokens"], _rows()[0]["output_tokens"]) == (7, 3)


def test_recorder_skips_reply_without_usage(caplog):
    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]], llm_output=None)
    metering.UsageRecorder("groq", "m").on_llm_end(result)
    assert _rows() == [] and "no token usage" in caplog.text


def test_recorder_swallows_ledger_errors(monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(metering, "record", boom)
    metering.UsageRecorder("groq", "m").on_llm_end(_streamed_result())  # must not raise


# --- Model calls (a real ChatGroq with a fake HTTP client) ----------------

def test_get_llm_attaches_exactly_one_recorder(groq_llm):
    model = groq_llm()
    assert sum(isinstance(cb, metering.UsageRecorder) for cb in model.callbacks) == 1


def test_one_invoke_writes_exactly_one_row(groq_llm, monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODEL", "openai/gpt-oss-120b")
    groq_llm(prompt_tokens=30, completion_tokens=4).invoke("hello")
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"], row["model"]) == (30, 4, "openai/gpt-oss-120b")
    assert row["cost_usd"] is not None


def test_async_invoke_is_recorded_once(groq_llm):
    asyncio.run(groq_llm(prompt_tokens=15, completion_tokens=3).ainvoke("hello"))
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"]) == (15, 3)


def test_json_schema_structured_output_is_recorded_once(groq_llm):
    class Out(BaseModel):
        score: int
    model = groq_llm(content='{"score": 7}', prompt_tokens=50, completion_tokens=6)
    assert model.with_structured_output(Out, method="json_schema").invoke("rate it").score == 7
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"]) == (50, 6)


def test_streamed_call_is_recorded_once(groq_llm):
    assert "".join(c.content for c in groq_llm(content="streamed", prompt_tokens=20, completion_tokens=2).stream("hi")) == "streamed"
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"]) == (20, 2)


def test_agent_turn_records_under_client_and_chat_feature(groq_llm):
    from langchain.agents import create_agent

    model = groq_llm(content="Hello!", prompt_tokens=80, completion_tokens=3)
    agent = create_agent(model=model, tools=[], system_prompt="Be brief.")

    async def go():
        with metering.context(client_id="acme", session_id="s1", feature="chat"):
            async for _ in agent.astream({"messages": [("user", "hi")]}, stream_mode="messages"):
                pass
    asyncio.run(go())
    # Streamed like main.py's agent turns, so usage came from the final x_groq chunk.
    assert [c.get("stream") for c in model.async_client.calls] == [True]
    (row,) = _rows()
    assert (row["client_id"], row["session_id"], row["feature"]) == ("acme", "s1", "chat")
    assert (row["input_tokens"], row["output_tokens"]) == (80, 3)


# --- Feature tags ---------------------------------------------------------

def test_guard_call_is_tagged_guard(groq_llm, monkeypatch):
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(llm, "get_llm", lambda **kw: groq_llm(content="CLEAN", **kw))
    with metering.context(client_id="acme", feature="chat"):
        assert guardrails.check_input_llm("How do I improve my CV?") == "CLEAN"
        assert metering.current()["feature"] == "chat"
    (row,) = _rows()
    assert (row["client_id"], row["feature"], row["model"]) == ("acme", "guard", config.GUARD_MODEL)


def test_interview_calls_are_tagged_by_step(groq_llm, monkeypatch):
    replies = iter([
        '{"role": "Python Developer", "job_summary": "Builds APIs.",'
        ' "questions": [{"text": "Tell me about an API you built.", "kind": "technical"}]}',
        '{"score": 7, "feedback": "Clear; add numbers."}',
        '{"strengths": ["Clear structure"], "improvements": ["Quantify impact"]}',
    ])
    monkeypatch.setattr(interview, "get_llm", lambda **kw: groq_llm(content=next(replies), **kw))
    with metering.context(client_id="acme", feature="chat"):
        qset = interview._generate_question_set("Senior Python role.", "", 1)
        ev = interview._evaluate_answer(qset.role, qset.job_summary, qset.questions[0].text, "I built one.")
        narrative = interview._report_narrative(
            qset.role, [{"question": qset.questions[0].text, "score": ev.score, "feedback": ev.feedback}])
        assert metering.current()["feature"] == "chat"
    assert qset.questions[0].kind == "technical" and ev.score == 7
    assert narrative.strengths == ["Clear structure"]
    assert [(r["client_id"], r["feature"]) for r in _rows()] == [
        ("acme", "interview.questions"), ("acme", "interview.score"), ("acme", "interview.report"),
    ]


# --- Client keys, allowances and the admin report (HTTP) ------------------

@pytest.fixture
def api(monkeypatch):
    """The app without its lifespan (so no MCP servers) and an empty session store."""
    import main
    import session_store
    monkeypatch.setattr(session_store, "_sessions", {})
    return TestClient(main.app)


def _start(api, key=None):
    res = api.post("/api/session", headers={"X-Client-Key": key} if key else {})
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def _send(api, sid, text):
    res = api.post("/api/message", json={"session_id": sid, "message": text})
    assert res.status_code == 200, res.text
    return [json.loads(line[len("data: "):]) for line in res.text.splitlines() if line.startswith("data: ")]


def _reply(frames):
    return "".join(f["text"] for f in frames if f["type"] == "token")


def _must_not_run(*args, **kwargs):
    raise AssertionError("no model may run once the allowance is used up")


def test_session_without_key_uses_default_client(api):
    import session_store
    sid = _start(api)
    assert session_store.get_session(sid).client_id == "default"
    assert session_store.get_session(sid).id == sid


def test_session_with_valid_key_is_bound_to_client(api):
    import session_store
    cid, key = metering.add_client("Acme", "starter")
    sid = _start(api, key)
    assert session_store.get_session(sid).client_id == cid


def test_session_with_unknown_key_is_rejected(api):
    import session_store
    assert api.post("/api/session", headers={"X-Client-Key": "ava_nope"}).status_code == 401
    assert session_store._sessions == {}


def test_session_requires_key_when_configured(api, monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_CLIENT_KEY", True)
    assert api.post("/api/session").status_code == 401
    _, key = metering.add_client("Acme", "starter")
    assert api.post("/api/session", headers={"X-Client-Key": key}).status_code == 200


def test_client_key_lookup_fails_open(api, monkeypatch, caplog):
    import session_store

    def boom(key):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(metering, "client_for_key", boom)
    sid = _start(api, "ava_anything")
    assert session_store.get_session(sid).client_id == "default"
    assert "key lookup failed" in caplog.text


def test_over_limit_streams_notice_without_calling_a_model(api, monkeypatch):
    import main
    _, key = metering.add_client("Acme", "suspended")
    sid = _start(api, key)
    monkeypatch.setattr(main, "guard_incoming", _must_not_run)
    monkeypatch.setattr(main, "stream_ava", _must_not_run)
    frames = _send(api, sid, "Please tailor my CV for a backend role")
    assert [f["type"] for f in frames] == ["token", "done"]
    assert "allowance" in frames[0]["text"]
    assert _rows() == []


def test_used_up_monthly_allowance_is_enforced(api, monkeypatch):
    import main
    cid, key = metering.add_client("Acme", "free")
    sid = _start(api, key)
    _record_for(cid, 100_000)
    monkeypatch.setattr(main, "guard_incoming", _must_not_run)
    monkeypatch.setattr(main, "stream_ava", _must_not_run)
    assert "allowance" in _reply(_send(api, sid, "Hello again"))


def test_over_limit_also_gates_interview_turns(api, monkeypatch):
    import main
    import session_store
    _, key = metering.add_client("Acme", "suspended")
    sid = _start(api, key)
    it = session_store.InterviewSession(
        id="iv1", role="Python Developer", job_summary="Builds APIs.",
        questions=[session_store.InterviewQuestion(text="Tell me about an API you built.", kind="technical")])
    session_store.get_session(sid).interview = it
    for name in ("guard_incoming", "handle_turn", "stream_ava"):
        monkeypatch.setattr(main, name, _must_not_run)
    frames = _send(api, sid, "I built a payments API in FastAPI.")
    assert [f["type"] for f in frames] == ["token", "done"]
    assert "allowance" in frames[0]["text"]
    assert it.current == 0 and it.results == []


def test_guard_call_is_billed_when_it_blocks(api, groq_llm, monkeypatch):
    import main
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(llm, "get_llm", lambda **kw: groq_llm(content="INJECTION", **kw))
    monkeypatch.setattr(main, "stream_ava", _must_not_run)
    cid, key = metering.add_client("Acme", "starter")
    sid = _start(api, key)
    # Passes the regex stages, so only the (faked) LLM guard blocks it.
    frames = _send(api, sid, "Please tell me about the weather today")
    assert "follow instructions embedded" in _reply(frames) and frames[-1]["type"] == "done"
    (row,) = _rows()
    assert (row["feature"], row["client_id"], row["session_id"]) == ("guard", cid, sid)
    assert row["model"] == config.GUARD_MODEL


@pytest.fixture
def fake_ava(groq_llm, monkeypatch):
    """The real agent loop on a faked ChatGroq that always answers "Hello from Ava"."""
    import agent
    from langchain.agents import create_agent
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", False)
    monkeypatch.setattr(agent, "_build_ava", lambda session: create_agent(
        model=groq_llm(content="Hello from Ava", prompt_tokens=60, completion_tokens=4),
        tools=[], system_prompt="x"))


def test_chat_turn_is_billed_to_the_session_client(api, fake_ava):
    cid, key = metering.add_client("Acme", "starter")
    sid = _start(api, key)
    frames = _send(api, sid, "Hi Ava")
    assert "Hello from Ava" in _reply(frames) and frames[-1]["type"] == "done"
    (row,) = _rows()
    assert (row["client_id"], row["session_id"], row["feature"]) == (cid, sid, "chat")
    assert (row["input_tokens"], row["output_tokens"]) == (60, 4)
    assert metering.check_allowance(cid).used == 64


def test_chat_still_streams_when_recording_fails(api, fake_ava, monkeypatch):
    sid = _start(api)

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(metering, "record", boom)
    assert "Hello from Ava" in _reply(_send(api, sid, "Hi Ava"))


def test_chat_still_streams_when_the_ledger_is_down(api, fake_ava, monkeypatch, caplog):
    sid = _start(api)

    def boom():
        raise sqlite3.OperationalError("unable to open database file")
    monkeypatch.setattr(metering, "_connect", boom)
    assert "Hello from Ava" in _reply(_send(api, sid, "Hi Ava"))
    assert "allowance check failed" in caplog.text


_ADMIN = {"X-Admin-Key": "s3cret-admin"}


def test_admin_usage_is_closed_without_an_admin_key(api, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "")
    assert api.get("/api/admin/usage").status_code == 404
    assert api.get("/api/admin/usage", headers=_ADMIN).status_code == 404


def test_admin_usage_requires_the_right_key(api, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "s3cret-admin")
    assert api.get("/api/admin/usage").status_code == 401
    assert api.get("/api/admin/usage", headers={"X-Admin-Key": "wrong"}).status_code == 401
    res = api.get("/api/admin/usage", headers=_ADMIN)
    assert res.status_code == 200
    body = res.json()
    assert body["month"] == metering.month_bounds()[0] and body["timezone"] == "UTC"
    assert isinstance(body["clients"], list) and isinstance(body["breakdown"], list)


def test_admin_usage_rejects_a_bad_month(api, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "s3cret-admin")
    assert api.get("/api/admin/usage", params={"month": "2026-13"}, headers=_ADMIN).status_code == 400


def test_admin_usage_filters_by_month_and_client(api, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "s3cret-admin")
    acme, _ = metering.add_client("Acme", "starter")
    beta, _ = metering.add_client("Beta", "free")
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 10, 1, client_id=acme)
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 20, 2, client_id=beta)
    body = api.get("/api/admin/usage", params={"month": "2026-10", "client_id": beta}, headers=_ADMIN).json()
    assert body["month"] == "2026-10"
    assert [(c["client_id"], c["used_tokens"]) for c in body["clients"]] == [(beta, 22)]
    assert [r["client_id"] for r in body["breakdown"]] == [beta]


def test_admin_usage_is_left_out_of_the_public_api_docs(api):
    import main
    assert "/api/admin/usage" in {getattr(route, "path", None) for route in main.app.routes}
    assert "/api/admin/usage" not in api.get("/openapi.json").json()["paths"]


def test_frontend_client_key_header_passes_cors(api):
    # The browser preflights the frontend's X-Client-Key header on session start.
    res = api.options("/api/session", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-client-key"})
    assert res.status_code == 200
    assert "x-client-key" in res.headers["access-control-allow-headers"].lower()


# --- Admin CLI (manage_clients.py) ------------------------------------------

def test_cli_add_list_set_plan_and_usage(capsys):
    assert manage_clients.main(["add", "Acme Ltd", "--plan", "starter"]) == 0
    out = capsys.readouterr().out
    key = next(w for w in out.split() if w.startswith("ava_"))
    cid = metering.client_for_key(key)
    assert cid and cid in out
    assert manage_clients.main(["list"]) == 0
    listed = capsys.readouterr().out
    # The key is printed once, when the client is created, and never again.
    assert cid in listed and "Acme Ltd" in listed and key not in listed
    assert hashlib.sha256(key.encode()).hexdigest() not in listed
    assert manage_clients.main(["set-plan", cid, "pro"]) == 0
    assert {c["client_id"]: c["plan"] for c in metering.list_clients()}[cid] == "pro"
    with metering.context(client_id=cid):
        metering.record("groq", "openai/gpt-oss-120b", 1000, 200)
    assert manage_clients.main(["usage"]) == 0
    usage_out = capsys.readouterr().out
    assert cid in usage_out and "1,200" in usage_out


def test_cli_rejects_unknown_plan(capsys):
    assert manage_clients.main(["add", "Acme", "--plan", "gold"]) == 2
    assert "unknown plan" in capsys.readouterr().err
    assert [c["client_id"] for c in metering.list_clients()] == ["default"]


def test_cli_plans_lists_plan_names(capsys):
    assert manage_clients.main(["plans"]) == 0
    out = capsys.readouterr().out
    assert "starter" in out and "uncapped" in out and "1,000,000" in out


def test_cli_set_plan_rejects_unknown_client(capsys):
    assert manage_clients.main(["set-plan", "nobody-000000", "pro"]) == 2
    assert "unknown client" in capsys.readouterr().err


def test_cli_usage_rejects_a_bad_month(capsys):
    assert manage_clients.main(["usage", "--month", "2026-13"]) == 2
    assert "YYYY-MM" in capsys.readouterr().err


def test_cli_usage_filters_by_month_and_client(monkeypatch, capsys):
    acme, _ = metering.add_client("Acme", "starter")
    beta, _ = metering.add_client("Beta", "free")
    _record_at(monkeypatch, "2026-09-30T23:59:59.000000Z", "openai/gpt-oss-120b", 300, 30, client_id=beta)
    _record_at(monkeypatch, "2026-10-01T00:00:00.000000Z", "openai/gpt-oss-120b", 40000, 4000, client_id=beta)
    _record_at(monkeypatch, "2026-09-15T12:00:00.000000Z", "openai/gpt-oss-120b", 5000, 500, client_id=acme)
    assert manage_clients.main(["usage", "--month", "2026-09", "--client", beta]) == 0
    out = capsys.readouterr().out
    assert "2026-09 (UTC)" in out and beta in out and "330" in out and "100,000" in out
    assert acme not in out and "44,000" not in out


def test_cli_usage_lists_unregistered_and_unpriced_usage(capsys):
    with metering.context(client_id="ghost"):
        metering.record("groq", "mystery-model", 70, 7)
    assert manage_clients.main(["usage"]) == 0
    lines = capsys.readouterr().out.splitlines()
    ghost = next(line for line in lines if line.startswith("ghost"))
    assert "77" in ghost and "n/a" in ghost and "1 unpriced call" in ghost
    # No calls at all is an exact $0, not an unpriced n/a.
    default = next(line for line in lines if line.startswith("default"))
    assert "uncapped" in default and "$0.0000" in default


def test_cli_runs_as_a_script(tmp_path):
    env = {**os.environ, "METERING_DB": str(tmp_path / "script.db")}
    done = subprocess.run([sys.executable, manage_clients.__file__, "list"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "default" in done.stdout
    assert (tmp_path / "script.db").exists()
