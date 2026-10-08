import asyncio
import contextvars
import hashlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import closing

import anyio
import groq
import httpx
import pytest
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from pydantic import BaseModel

import config
import guardrails
import interview
import llm
import manage_clients
import metering


def _rows():
    metering.flush()  # rows from model calls reach the ledger through a writer thread
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
    assert (second["client_id"], second["session_id"], second["feature"]) == ("acme", metering.session_ref("s1"), "chat")
    assert second["total_tokens"] == 15 and second["provider"] == "groq"
    assert first["ts_utc"].endswith("Z")
    assert first["estimated"] == second["estimated"] == 0


def test_ledger_stores_no_message_text():
    with metering.context(client_id="acme", session_id="s1", feature="chat"):
        metering.record("groq", "openai/gpt-oss-120b", 10, 5)
    assert set(_rows()[0]) == {
        "id", "ts_utc", "client_id", "session_id", "feature", "provider", "model",
        "input_tokens", "output_tokens", "total_tokens", "cost_usd", "estimated",
    }


def test_ledger_from_before_estimates_is_upgraded_in_place(monkeypatch):
    # A ledger written by an earlier version has no `estimated` column. It is
    # added on first use, its rows count as exact, and nothing is lost.
    old = sqlite3.connect(config.METERING_DB)
    old.executescript("""
        CREATE TABLE clients (client_id TEXT PRIMARY KEY, name TEXT NOT NULL, plan TEXT NOT NULL,
                              key_hash TEXT UNIQUE, created_utc TEXT NOT NULL);
        CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL,
                            client_id TEXT NOT NULL, session_id TEXT, feature TEXT NOT NULL,
                            provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL,
                            output_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL, cost_usd REAL);
        INSERT INTO clients VALUES ('acme-123456', 'Acme', 'free', 'somehash', '2026-10-01T00:00:00.000000Z');
        INSERT INTO usage (ts_utc, client_id, session_id, feature, provider, model, input_tokens,
                           output_tokens, total_tokens, cost_usd)
        VALUES ('2026-10-06T09:00:00.000000Z', 'acme-123456', NULL, 'chat', 'groq',
                'openai/gpt-oss-120b', 100, 20, 120, 0.000027);
    """)
    old.close()
    _record_at(monkeypatch, "2026-10-06T10:00:00.000000Z", "openai/gpt-oss-120b", 10, 5,
               client_id="acme-123456")
    metering.record("groq", "openai/gpt-oss-120b", 1, 1, estimated=True)
    assert [(r["total_tokens"], r["estimated"]) for r in _rows()] == [(120, 0), (15, 0), (2, 1)]
    (acme,) = [c for c in metering.usage_report("2026-10")["clients"] if c["client_id"] == "acme-123456"]
    assert acme["used_tokens"] == 135 and acme["estimated_calls"] == 0


def test_ledger_upgrade_tolerates_another_process_adding_the_column_first(monkeypatch):
    # The server and the CLI (or two workers) open a pre-upgrade ledger at the
    # same moment: both see the column missing, and the other one adds it first.
    old = sqlite3.connect(config.METERING_DB)
    old.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL, "
                "client_id TEXT NOT NULL, session_id TEXT, feature TEXT NOT NULL, provider TEXT NOT NULL, "
                "model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
                "total_tokens INTEGER NOT NULL, cost_usd REAL)")
    old.close()
    real_connect = sqlite3.connect

    class OtherProcessWinsTheRace:
        def __init__(self, conn):
            self._conn = conn
            self._raced = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, *args):
            result = self._conn.execute(sql, *args)
            if sql.startswith("PRAGMA table_info") and not self._raced:
                self._raced = True
                columns = result.fetchall()   # what this process saw: no column yet
                with closing(real_connect(config.METERING_DB)) as other:
                    other.execute("ALTER TABLE usage ADD COLUMN estimated INTEGER NOT NULL DEFAULT 0")
                return iter(columns)
            return result

    connects = []

    def connect(*args, **kwargs):
        connects.append(args)
        conn = real_connect(*args, **kwargs)
        return OtherProcessWinsTheRace(conn) if len(connects) == 1 else conn
    monkeypatch.setattr(metering.sqlite3, "connect", connect)
    metering.record("groq", "openai/gpt-oss-120b", 10, 5, estimated=True)
    assert [(r["total_tokens"], r["estimated"]) for r in _rows()] == [(15, 1)]


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


@pytest.mark.parametrize("content", ['{"openai/gpt-oss-120b": {"input": 0.15, "output": 0.6},}', "[]"])
def test_unreadable_pricing_file_still_records_tokens(tmp_path, monkeypatch, caplog, content):
    # A stray comma in the hand-edited price list must cost us prices, not rows.
    bad = tmp_path / "pricing.json"
    bad.write_text(content, encoding="utf-8")
    monkeypatch.setattr(config, "PRICING_FILE", str(bad))
    monkeypatch.setattr(metering, "_prices", None)
    monkeypatch.setattr(metering, "_warned_models", set())
    with caplog.at_level(logging.ERROR, logger="metering"):
        with metering.context(client_id="acme", feature="chat"):
            metering.UsageRecorder("groq", "openai/gpt-oss-120b").on_llm_end(_streamed_result(40, 9))
            metering.UsageRecorder("groq", "openai/gpt-oss-120b").on_llm_end(_streamed_result(40, 9))
    assert [(r["client_id"], r["total_tokens"], r["cost_usd"]) for r in _rows()] == [("acme", 49, None)] * 2
    # Read (and reported) once, not again on every call.
    assert sum("could not load" in r.getMessage() for r in caplog.records) == 1


def test_unreadable_plans_file_leaves_clients_uncapped(tmp_path, monkeypatch):
    client_id, _ = metering.add_client("Acme", "suspended")
    bad = tmp_path / "plans.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(config, "PLANS_FILE", str(bad))
    monkeypatch.setattr(metering, "_plans", None)
    _record_for(client_id, 500)
    assert metering.check_allowance(client_id) == metering.Allowance(True, 500, None)
    (acme,) = [c for c in metering.usage_report()["clients"] if c["client_id"] == client_id]
    assert acme["used_tokens"] == 500 and acme["monthly_tokens"] is None


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


def test_client_cost_is_null_when_any_call_is_unpriced(monkeypatch, capsys):
    # A partial sum would bill every unpriced call as $0.
    acme, _ = metering.add_client("Acme", "starter")
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-20b", 10, 5, client_id=acme)
    _record_at(monkeypatch, "2026-10-06T10:00:00.000000Z", "mystery-model", 50, 5, client_id=acme)
    (client,) = metering.usage_report("2026-10", client_id=acme)["clients"]
    assert (client["cost_usd"], client["unpriced_calls"], client["used_tokens"]) == (None, 1, 70)
    assert manage_clients.main(["usage", "--month", "2026-10", "--client", acme]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith(acme))
    assert "n/a" in line and "$" not in line and "1 unpriced call" in line


def test_usage_report_rejects_bad_month():
    with pytest.raises(ValueError):
        metering.usage_report("2026-13")


def test_usage_report_counts_estimated_calls(monkeypatch):
    acme, _ = metering.add_client("Acme", "starter")
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "openai/gpt-oss-120b", 100, 10,
               client_id=acme, feature="chat")
    with metering.context(client_id=acme, feature="chat"):
        for _ in range(2):
            metering.record("groq", "openai/gpt-oss-120b", 40, 6, estimated=True)
    report = metering.usage_report("2026-10")
    (row,) = [r for r in report["breakdown"] if r["client_id"] == acme]
    assert (row["calls"], row["estimated_calls"], row["total_tokens"]) == (3, 2, 202)
    clients = {c["client_id"]: c for c in report["clients"]}
    assert clients[acme]["estimated_calls"] == 2 and clients["default"]["estimated_calls"] == 0
    assert "cut off" in report["note"] and "estimated=1" in report["note"]


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
    metering.flush()  # nor the writer thread: the failure is only logged


def _chunk(text="", **message):
    return ChatGenerationChunk(message=AIMessageChunk(content=text, **message))


def _open_call(recorder):
    """Start a call on the recorder and stream part of a reply, as LangChain would."""
    run_id = uuid.uuid4()
    recorder.on_chat_model_start({}, [[HumanMessage("Tailor my CV for a data role")]], run_id=run_id)
    recorder.on_llm_new_token("Sure, ", chunk=_chunk("Sure, "), run_id=run_id)
    # Reasoning and tool-call arguments are output Groq bills too.
    recorder.on_llm_new_token("", chunk=_chunk(additional_kwargs={"reasoning_content": "Check the CV. "}),
                              run_id=run_id)
    recorder.on_llm_new_token("", chunk=_chunk(tool_call_chunks=[
        {"name": "search_cv", "args": '{"query": "data"}', "id": "c1", "index": 0}]), run_id=run_id)
    return run_id


@pytest.mark.parametrize("error", [asyncio.CancelledError(), GeneratorExit()], ids=["cancelled", "closed"])
def test_recorder_estimates_a_call_cut_off_mid_reply(error):
    recorder = metering.UsageRecorder("groq", "openai/gpt-oss-120b")
    with metering.context(client_id="acme", session_id="s1", feature="chat"):
        run_id = _open_call(recorder)
    # Attribution is read when the call starts, not where the error lands.
    recorder.on_llm_error(error, run_id=run_id)
    (row,) = _rows()
    assert (row["client_id"], row["session_id"], row["feature"]) == ("acme", metering.session_ref("s1"), "chat")
    assert row["estimated"] == 1 and row["cost_usd"] is not None
    # About four characters a token, rounded up.
    assert row["input_tokens"] == math.ceil(len("Human: Tailor my CV for a data role") / 4)
    assert row["output_tokens"] == math.ceil(len('Sure, Check the CV. {"query": "data"}') / 4)
    assert recorder._calls == {}


def test_recorder_uses_exact_usage_if_it_arrived_before_the_cut():
    recorder = metering.UsageRecorder("groq", "openai/gpt-oss-120b")
    run_id = _open_call(recorder)
    recorder.on_llm_new_token("", chunk=_chunk(usage_metadata={"input_tokens": 300, "output_tokens": 25,
                                                               "total_tokens": 325}), run_id=run_id)
    recorder.on_llm_error(asyncio.CancelledError(), run_id=run_id)
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (300, 25, 0)


def _rate_limited():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.RateLimitError("Rate limit reached", response=httpx.Response(429, request=request), body=None)


def test_recorder_records_nothing_for_a_failed_call():
    # Groq doesn't bill a request that fails (rate limit, 5xx), even part-way through.
    recorder = metering.UsageRecorder("groq", "openai/gpt-oss-120b")
    recorder.on_llm_error(_rate_limited(), run_id=_open_call(recorder))
    assert _rows() == [] and recorder._calls == {}


def test_recorder_records_nothing_for_a_call_cut_off_before_groq_answered():
    # No chunk yet means Groq may never have accepted the request (the client
    # was still connecting, or waiting to retry a rate limit), so nothing is billed.
    recorder = metering.UsageRecorder("groq", "openai/gpt-oss-120b")
    run_id = uuid.uuid4()
    recorder.on_chat_model_start({}, [[HumanMessage("Tailor my CV for a data role")]], run_id=run_id)
    recorder.on_llm_error(asyncio.CancelledError(), run_id=run_id)
    assert _rows() == [] and recorder._calls == {}


def test_call_cancelled_while_waiting_to_retry_a_rate_limit_is_not_recorded(monkeypatch):
    # A real ChatGroq and groq client; only the HTTP transport is faked. Groq
    # answers 429 and the client sleeps before retrying, as it does in a rate
    # limit spell. The user gives up during that sleep: Groq billed nothing.
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    sent = []

    def rate_limited(request):
        sent.append(request)
        return httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "Rate limit reached"}})
    model = llm.get_llm()
    model.async_client = groq.AsyncGroq(
        api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(rate_limited)),
    ).chat.completions

    async def go():
        with metering.context(client_id="acme", feature="chat"):
            with anyio.move_on_after(0.5):
                async for _ in model.astream("Hi Ava"):
                    pass
    anyio.run(go)
    assert len(sent) == 1
    assert _rows() == [] and model.callbacks[0]._calls == {}


# --- Model calls (a real ChatGroq with a fake HTTP client) ----------------

def test_get_llm_attaches_exactly_one_recorder(groq_llm):
    model = groq_llm()
    assert sum(isinstance(cb, metering.UsageRecorder) for cb in model.callbacks) == 1


def test_one_invoke_writes_exactly_one_row(groq_llm, monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODEL", "openai/gpt-oss-120b")
    groq_llm(prompt_tokens=30, completion_tokens=4).invoke("hello")
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"], row["model"]) == (30, 4, "openai/gpt-oss-120b")
    assert row["cost_usd"] is not None and row["estimated"] == 0


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
    model = groq_llm(content=["str", "eam", "ed"], prompt_tokens=20, completion_tokens=2)
    assert "".join(c.content for c in model.stream("hi")) == "streamed"
    (row,) = _rows()
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (20, 2, 0)
    assert model.callbacks[0]._calls == {}


def test_stream_closed_early_is_recorded_as_an_estimate(groq_llm):
    # The caller stops reading: LangChain reports GeneratorExit through on_llm_error.
    model = groq_llm(content=["Hello ", "there"], prompt_tokens=20, completion_tokens=2)
    stream = model.stream("hi")
    assert next(stream).content == "Hello "
    stream.close()
    (row,) = _rows()
    assert (row["output_tokens"], row["estimated"]) == (math.ceil(len("Hello ") / 4), 1)
    assert 0 < row["input_tokens"] < 20


def test_async_stream_cancelled_in_a_cancel_scope_is_recorded_as_an_estimate(groq_llm):
    # anyio cancellation is level-triggered, as Starlette delivers it: every
    # await on the way out is cancelled again, including LangChain's await on
    # its own error callback. The row must be written anyway.
    model = groq_llm(content=["Hello ", "there, ", "I can ", "help"], prompt_tokens=60, completion_tokens=4)
    model.async_client.before_usage = asyncio.Event().wait  # the rest of the reply never comes

    async def go():
        with metering.context(client_id="acme", feature="chat"):
            with anyio.CancelScope() as scope:
                async for _ in model.astream("Hi Ava"):
                    scope.cancel()
    anyio.run(go)
    (row,) = _rows()
    assert (row["client_id"], row["feature"], row["estimated"]) == ("acme", "chat", 1)
    assert row["output_tokens"] == math.ceil(len("Hello there, I can help") / 4)
    assert model.callbacks[0]._calls == {}


def test_locked_ledger_does_not_freeze_the_event_loop(groq_llm):
    # Another writer (e.g. a DB browser with an unsaved edit) holds the ledger's
    # write lock. Recording must wait for it off the event loop (in the writer
    # thread), so other chats keep streaming and the holder can finish; on the
    # loop, it would freeze for the whole busy timeout and the row would be lost.
    model = groq_llm(prompt_tokens=15, completion_tokens=3)
    asyncio.run(model.ainvoke("warm up"))  # one-off client setup, before the clock starts
    metering.flush()  # and the ledger exists, as it does once a turn has checked its allowance
    holder = sqlite3.connect(config.METERING_DB, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")

    async def go():
        loop = asyncio.get_running_loop()
        gaps, running = [], True

        async def heartbeat():
            last = loop.time()
            while running:
                await asyncio.sleep(0.02)
                gaps.append(loop.time() - last)
                last = loop.time()
        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)
        # Only a running loop can let the other writer finish.
        loop.call_later(0.3, holder.execute, "COMMIT")
        with metering.context(client_id="acme", session_id="s1", feature="chat"):
            await model.ainvoke("hello")
        running = False
        await beat
        return max(gaps)
    try:
        longest_gap = asyncio.run(go())
    finally:
        holder.close()
    assert longest_gap < 1.0
    row = _rows()[-1]
    assert (row["client_id"], row["feature"], row["input_tokens"]) == ("acme", "chat", 15)


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
    assert (row["client_id"], row["session_id"], row["feature"]) == ("acme", metering.session_ref("s1"), "chat")
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (80, 3, 0)


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


def test_client_key_lookup_failure_is_503_when_keys_are_required(api, monkeypatch):
    # With keys required, a chat must never land on the uncapped default client
    # just because the ledger couldn't be read.
    import session_store
    monkeypatch.setattr(config, "REQUIRE_CLIENT_KEY", True)

    def boom(key):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(metering, "client_for_key", boom)
    res = api.post("/api/session", headers={"X-Client-Key": "ava_anything"})
    assert res.status_code == 503
    assert res.json()["detail"] == "Usage service unavailable, please try again shortly."
    assert session_store._sessions == {}


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


def _allow_all(message):
    return guardrails.GuardResult(allowed=True, cleaned_message=message, category=None,
                                  safe_reply=None, dialect="en")


async def _turn_frames(session, text):
    import main
    return [json.loads(f[len("data: "):]) async for f in main._message_events(session, text)]


def test_parallel_turns_started_together_may_all_finish(monkeypatch):
    # One check per turn, before any model call, and a turn that has started may
    # finish: turns that pass the check together can each run to the end and
    # take a client slightly over its allowance. The next turn is then refused.
    import main
    import session_store
    cid, _ = metering.add_client("Acme", "free")
    _record_for(cid, 99_000)
    sessions = [session_store.Session(id=f"s{i}", client_id=cid) for i in range(3)]
    monkeypatch.setattr(main, "guard_incoming", _allow_all)
    started = []

    async def turn(session, message):
        started.append(session.id)
        while len(started) < len(sessions):  # all three are past the check
            await asyncio.sleep(0.01)
        metering.record("groq", "openai/gpt-oss-120b", 5_000, 0)
        yield ("token", "Done.")
        yield ("done", None)
    monkeypatch.setattr(main, "stream_ava", turn)

    async def burst():
        turns = asyncio.gather(*(_turn_frames(s, "Find me jobs") for s in sessions))
        return await asyncio.wait_for(turns, 5)
    assert [_reply(frames) for frames in asyncio.run(burst())] == ["Done."] * 3
    assert metering.check_allowance(cid).used == 99_000 + 15_000
    assert "allowance" in _reply(asyncio.run(_turn_frames(sessions[0], "One more")))


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


def test_interview_answer_is_billed_to_the_session_client(api, groq_llm, monkeypatch):
    import session_store
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", False)
    monkeypatch.setattr(interview, "get_llm", lambda **kw: groq_llm(
        content='{"score": 7, "feedback": "Clear; add numbers."}', prompt_tokens=50, completion_tokens=6, **kw))
    cid, key = metering.add_client("Acme", "starter")
    sid = _start(api, key)
    session_store.get_session(sid).interview = session_store.InterviewSession(
        id="iv1", role="Python Developer", job_summary="Builds APIs.",
        questions=[session_store.InterviewQuestion(text="Tell me about an API you built.", kind="technical"),
                   session_store.InterviewQuestion(text="How do you test it?", kind="technical")])
    # Scored off the event loop: the client and session must survive the hop.
    frames = _send(api, sid, "I built a payments API in FastAPI.")
    assert "7/10" in _reply(frames) and frames[-1]["type"] == "done"
    (row,) = _rows()
    assert (row["client_id"], row["session_id"], row["feature"]) == (cid, metering.session_ref(sid), "interview.score")
    assert metering.check_allowance(cid).used == 56


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
    assert (row["feature"], row["client_id"], row["session_id"]) == ("guard", cid, metering.session_ref(sid))
    assert row["model"] == config.GUARD_MODEL


def _guard_answers(fake, clean, injection):
    """Clef's answers to the guard's questions: a choice and three yes/no."""
    return {"guard": fake.choice({"clean": clean, "injection": injection, "abuse": 0.0, "harmful": 0.0}),
            "is_injection": fake.noul(injection), "is_abuse": fake.noul(0.0), "is_harmful": fake.noul(0.0)}


def test_clef_guard_call_is_billed_when_it_blocks(api, clef_api, monkeypatch):
    import main
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(config, "GUARD_BACKEND", "clef")
    monkeypatch.setattr(guardrails, "_groq_guard", _must_not_run)
    monkeypatch.setattr(main, "stream_ava", _must_not_run)
    clef_api.reply(_guard_answers(clef_api, clean=0.1, injection=0.9))
    cid, key = metering.add_client("Acme", "starter")
    sid = _start(api, key)
    frames = _send(api, sid, "Please tell me about the weather today")
    assert "follow instructions embedded" in _reply(frames) and frames[-1]["type"] == "done"
    (row,) = _rows()
    assert (row["provider"], row["model"], row["feature"]) == ("cloudflare", "@cf/cloudflare/clef", "guard")
    assert (row["client_id"], row["session_id"], row["input_tokens"]) == (cid, metering.session_ref(sid), 346)


def test_shadow_guard_never_delays_the_reply(api, clef_api, groq_llm, monkeypatch):
    # Groq decides and the reply streams at once; Clef, held up here for 5 s,
    # answers in the background and is billed to the same client and chat.
    import main
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(config, "GUARD_BACKEND", "shadow")
    monkeypatch.setattr(llm, "get_llm", lambda **kw: groq_llm(content="CLEAN", **kw))
    clef_free = threading.Event()
    clef_api.before = lambda body: clef_free.wait(5)
    clef_api.reply(_guard_answers(clef_api, clean=0.97, injection=0.03))

    async def reply_at_once(session, message):
        yield ("token", "Hello from Ava")
        yield ("done", None)
    monkeypatch.setattr(main, "stream_ava", reply_at_once)
    cid, key = metering.add_client("Acme", "starter")
    sid = _start(api, key)
    started = time.perf_counter()
    frames = _send(api, sid, "How do I improve my CV?")
    assert _reply(frames) == "Hello from Ava" and time.perf_counter() - started < 2
    assert [r["feature"] for r in _rows()] == ["guard"]   # Clef hasn't answered yet
    clef_free.set()
    assert guardrails._shadow_drain(5)
    ref = metering.session_ref(sid)
    assert [(r["provider"], r["feature"], r["client_id"], r["session_id"]) for r in _rows()] == [
        ("groq", "guard", cid, ref), ("cloudflare", "guard.shadow", cid, ref)]


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
    assert (row["client_id"], row["session_id"], row["feature"]) == (cid, metering.session_ref(sid), "chat")
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (60, 4, 0)
    assert metering.check_allowance(cid).used == 64


def _agent_on(monkeypatch, model):
    """Ava's turns run the real agent loop on this (faked) model."""
    import agent
    from langchain.agents import create_agent
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", False)
    monkeypatch.setattr(agent, "_build_ava", lambda session: create_agent(model=model, tools=[], system_prompt="x"))


def test_reply_cut_off_mid_stream_is_recorded_as_an_estimate(groq_llm, monkeypatch):
    # The user closes the tab while Ava is still replying. Starlette cancels the
    # response (an anyio cancel scope) while it waits to send, then the reply
    # it abandoned is closed, which cancels the model call still streaming.
    # LangChain reports nothing for that call, but Groq bills what it generated.
    import main
    import session_store
    model = groq_llm(content=["Hello ", "there, ", "I can ", "help"], prompt_tokens=60, completion_tokens=4)
    model.async_client.before_usage = asyncio.Event().wait  # the rest of the reply never comes
    _agent_on(monkeypatch, model)
    cid, _ = metering.add_client("Acme", "starter")
    session = session_store.Session(id="s1", client_id=cid)

    async def go():
        turn = main._message_events(session, "Hi Ava")
        with anyio.CancelScope() as scope:
            async for frame in turn:
                if '"token"' in frame:
                    scope.cancel()
                    await anyio.sleep(5)  # Starlette's `await send(...)`, cancelled
        await turn.aclose()
        for _ in range(100):  # the cut-off call is recorded as its task unwinds
            if rows := _rows():
                return rows
            await asyncio.sleep(0.02)
        return []
    (row,) = anyio.run(go)
    assert (row["client_id"], row["session_id"], row["feature"]) == (cid, metering.session_ref("s1"), "chat")
    assert row["estimated"] == 1
    assert row["output_tokens"] == math.ceil(len("Hello there, I can help") / 4)
    assert 0 < row["input_tokens"] < 20
    assert metering.check_allowance(cid).used == row["total_tokens"]
    assert model.callbacks[0]._calls == {}


def test_reply_still_generating_when_the_user_leaves_is_billed(groq_llm, monkeypatch):
    # Starlette's own disconnect path, through the real app. Here the model
    # call is left to finish in the background, so its count is exact.
    import main
    import session_store
    model = groq_llm(content=["Hello ", "there"], prompt_tokens=60, completion_tokens=4)
    model.async_client.before_usage = lambda: asyncio.sleep(0.1)
    _agent_on(monkeypatch, model)
    monkeypatch.setattr(session_store, "_sessions", {})
    cid, _ = metering.add_client("Acme", "starter")
    sid, _ = session_store.create_session(client_id=cid)

    async def go():
        body = json.dumps({"session_id": sid, "message": "Hi Ava"}).encode()
        requested, left = [], anyio.Event()

        async def receive():
            if not requested:
                requested.append(True)
                return {"type": "http.request", "body": body, "more_body": False}
            await left.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and b'"token"' in message.get("body", b""):
                left.set()  # the user closes the tab after the first words
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/api/message", "raw_path": b"/api/message",
                 "query_string": b"", "root_path": "", "client": ("127.0.0.1", 1), "server": ("test", 80),
                 "headers": [(b"host", b"test"), (b"content-type", b"application/json")]}
        await main.app(scope, receive, send)
        for _ in range(100):
            if rows := _rows():
                return rows
            await asyncio.sleep(0.02)
        return []
    (row,) = anyio.run(go)
    assert (row["client_id"], row["session_id"]) == (cid, metering.session_ref(sid))
    assert (row["input_tokens"], row["output_tokens"], row["estimated"]) == (60, 4, 0)


def test_failed_model_call_in_a_turn_is_not_recorded(groq_llm, monkeypatch):
    import session_store
    model = groq_llm(content=["Hello ", "there"], prompt_tokens=60, completion_tokens=4)

    async def rate_limited():
        raise _rate_limited()
    model.async_client.before_usage = rate_limited
    _agent_on(monkeypatch, model)
    session = session_store.Session(id="s1", client_id="acme")
    frames = asyncio.run(_turn_frames(session, "Hi Ava"))
    assert frames[-1]["type"] == "done"  # Ava apologises instead
    assert _rows() == [] and model.callbacks[0]._calls == {}


def test_ledger_never_holds_a_usable_session_id(api, fake_ava):
    import session_store
    first, second = _start(api), _start(api)
    for sid in (first, first, second):
        _send(api, sid, "Hi Ava")
    a1, a2, b = (r["session_id"] for r in _rows())
    # A session id alone opens a chat (and the CV uploaded to it), so the ledger
    # keeps a one-way reference: enough to group a chat's calls, useless as a key.
    assert a1 == a2 != b
    assert {a1, b}.isdisjoint({first, second})
    assert session_store.get_session(a1) is None and session_store.get_session(b) is None


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
    # Every method looks like a missing page; a 405 would give the route away.
    for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"):
        res = api.request(method, "/api/admin/usage", headers=_ADMIN)
        assert res.status_code == 404, method
        assert res.json() == api.get("/api/no-such-route").json()
        assert "allow" not in res.headers
    # Nor may a trailing slash redirect to it, where a missing page gets a 404.
    for method in ("GET", "POST", "DELETE"):
        res = api.request(method, "/api/admin/usage/", headers=_ADMIN, follow_redirects=False)
        assert res.status_code == 404, method
        assert res.json() == api.get("/api/no-such-route").json()


def test_admin_usage_only_answers_get_when_open(api, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "s3cret-admin")
    for method in ("POST", "DELETE"):
        assert api.request(method, "/api/admin/usage", headers=_ADMIN).status_code in (404, 405)


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


def test_admin_usage_reports_zero_cost_for_clients_without_calls(api, monkeypatch):
    # null cost means "unpriced"; a client that made no calls cost exactly $0.
    monkeypatch.setattr(config, "ADMIN_API_KEY", "s3cret-admin")
    acme, _ = metering.add_client("Acme", "starter")
    _record_at(monkeypatch, "2026-10-06T09:00:00.000000Z", "mystery-model", 10, 1, client_id="ghost")
    body = api.get("/api/admin/usage", params={"month": "2026-10"}, headers=_ADMIN).json()
    clients = {c["client_id"]: c for c in body["clients"]}
    assert (clients[acme]["cost_usd"], clients[acme]["unpriced_calls"]) == (0.0, 0)
    assert clients["default"]["cost_usd"] == 0.0
    assert (clients["ghost"]["cost_usd"], clients["ghost"]["unpriced_calls"]) == (None, 1)


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


def test_cli_usage_shows_estimated_calls(capsys):
    with metering.context(client_id="ghost"):
        metering.record("groq", "openai/gpt-oss-120b", 70, 7)
        metering.record("groq", "openai/gpt-oss-120b", 30, 3, estimated=True)
    assert manage_clients.main(["usage"]) == 0
    lines = capsys.readouterr().out.splitlines()
    ghost = next(line for line in lines if line.startswith("ghost"))
    assert "110" in ghost and "(1 estimated call)" in ghost
    assert "estimated" not in next(line for line in lines if line.startswith("default"))


def test_app_shutdown_writes_queued_usage(monkeypatch):
    import main

    async def no_mcp():
        pass
    monkeypatch.setattr(main, "init_mcp", no_mcp)
    flushed = []
    monkeypatch.setattr(metering, "flush", lambda: flushed.append(True))

    async def go():
        async with main._lifespan(main.app):
            assert flushed == []
    asyncio.run(go())
    assert flushed == [True]


def test_queued_usage_is_written_at_interpreter_exit(tmp_path):
    # The process exits straight after the call ends; the row must not be lost.
    script = (
        "import metering\n"
        "from langchain_core.messages import AIMessageChunk\n"
        "from langchain_core.outputs import ChatGenerationChunk, LLMResult\n"
        "msg = AIMessageChunk(content='hi', usage_metadata={'input_tokens': 7, 'output_tokens': 2, 'total_tokens': 9})\n"
        "metering.UsageRecorder('groq', 'm').on_llm_end(LLMResult(generations=[[ChatGenerationChunk(message=msg)]]))\n"
    )
    env = {**os.environ, "METERING_DB": str(tmp_path / "exit.db")}
    done = subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(metering.__file__), env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    conn = sqlite3.connect(tmp_path / "exit.db")
    try:
        assert conn.execute("SELECT total_tokens FROM usage").fetchall() == [(9,)]
    finally:
        conn.close()


def test_cli_runs_as_a_script(tmp_path):
    env = {**os.environ, "METERING_DB": str(tmp_path / "script.db")}
    done = subprocess.run([sys.executable, manage_clients.__file__, "list"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "default" in done.stdout
    assert (tmp_path / "script.db").exists()
