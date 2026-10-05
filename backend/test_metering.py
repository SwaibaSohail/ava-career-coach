import asyncio
import contextvars
import hashlib
import logging
import sqlite3

import pytest
from fastapi.concurrency import run_in_threadpool

import config
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
