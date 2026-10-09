"""Cloudflare Clef client: request shape, local limits, parsing, error kinds,
the daily-quota pause, logging and metering. Hermetic: every call goes to the
clef_api fake in conftest.py, and an un-faked call fails the test."""

import asyncio
import copy
import importlib.util
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import pytest

import clef
import config
import metering

STATE = "Checkout has been failing for every customer for the last hour."

QUESTIONS = {
    "urgent": clef.Noul("Is this support request urgent?"),
    "team": clef.Choice("Which team should handle this request?", {
        "billing": "Payments, invoices, and refunds",
        "technical": "Outages, errors, and configuration",
        "sales": "Plans and upgrades",
    }),
    "severity": clef.Score("How severe is the customer impact?", ["No impact", "Minor", "Major", "Critical"]),
}


def _answers(fake):
    return {
        "urgent": fake.noul(0.83),
        "team": fake.choice({"billing": 0.05, "technical": 0.9, "sales": 0.05}),
        "severity": fake.score([0.05, 0.15, 0.6, 0.2], QUESTIONS["severity"].criteria),
    }


# A live clef-flash response, verbatim (research brief, section 2.3).
LIVE_RESPONSE = {
    "result": {
        "model": "clef-flash",
        "answers": {
            "category": {
                "type": "choice",
                "choice": "bug_report",
                "probabilities": {"bug_report": 0.9664, "feature_request": 0.0135, "billing": 0.0058, "other": 0.0143},
                "confidence": 0.9126,
            },
            "severity": {
                "type": "score",
                "score": 1.1069,
                "legend": {"0": "Cosmetic; no impact on functionality",
                           "1": "Broken or degraded feature, but a workaround exists",
                           "2": "Blocking issue; no workaround exists"},
                "probabilities": {"0": 0.0636, "1": 0.766, "2": 0.1704},
                "confidence": 0.4298,
            },
            "has_repro_steps": {"type": "noul", "noul": 0.5355},
        },
        "usage": {"input_tokens": 395, "output_tokens": 0},
    },
    "success": True,
    "errors": [],
    "messages": [],
}

LIVE_QUESTIONS = {
    "category": clef.Choice("What kind of ticket is this?", {
        "bug_report": "Something is broken", "feature_request": "Something new is wanted",
        "billing": "Payments and invoices", "other": "Anything else",
    }),
    "severity": clef.Score("How severe is it?", [
        "Cosmetic; no impact on functionality",
        "Broken or degraded feature, but a workaround exists",
        "Blocking issue; no workaround exists",
    ]),
    "has_repro_steps": clef.Noul("Does it include steps to reproduce?"),
}


def _rows():
    metering.flush()  # rows reach the ledger through a writer thread
    conn = sqlite3.connect(config.METERING_DB)
    conn.row_factory = sqlite3.Row
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'usage'").fetchone():
            return []
        return [dict(r) for r in conn.execute("SELECT * FROM usage ORDER BY id")]
    finally:
        conn.close()


def _fails(kind, *args, **kwargs):
    with pytest.raises(clef.ClefError) as err:
        clef.decide(*args, **kwargs)
    assert err.value.kind == kind
    return err.value


# --- Request shape --------------------------------------------------------

@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_request_goes_to_workers_ai_with_a_bearer_token(clef_api, model):
    clef_api.reply(_answers(clef_api))
    decision = clef.decide(STATE, QUESTIONS, model=model)
    (request,) = clef_api.requests
    assert request["url"] == f"https://api.cloudflare.com/client/v4/accounts/acct-test/ai/run/@cf/cloudflare/{model}"
    assert request["headers"]["authorization"] == "Bearer tok-test"
    assert request["headers"]["content-type"] == "application/json"
    # Only the fields the hosted schema takes; the body's model is the URL's short name.
    assert set(request["json"]) == {"model", "state", "questions"}
    assert (request["json"]["model"], request["json"]["state"]) == (model, STATE)
    assert decision.model == clef.MODELS[model] == f"@cf/cloudflare/{model}"


def test_questions_are_serialised_by_type():
    body = clef.build_body("clef", {"message": "hi"}, {
        "urgent": clef.Noul("Is it urgent?"),
        "spam": clef.Noul("Is it spam?", {"true": "Unsolicited", "false": "Anything else"}),
        "team": clef.Choice("Which team?", {"billing": "Payments", "technical": "Outages"}),
        "severity": clef.Score("How severe?", ("None", "Minor", "Major")),
    })
    assert body == {"model": "clef", "state": {"message": "hi"}, "questions": {
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "spam": {"type": "noul", "instructions": "Is it spam?",
                 "criteria": {"true": "Unsolicited", "false": "Anything else"}},
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "Payments", "technical": "Outages"}},
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["None", "Minor", "Major"]},
    }}


def test_the_built_body_is_what_is_sent(clef_api):
    clef_api.reply(_answers(clef_api))
    clef.decide(STATE, QUESTIONS)
    assert clef_api.requests[0]["json"] == clef.build_body("clef", STATE, QUESTIONS)


def test_timeout_comes_from_config_unless_given(clef_api, monkeypatch):
    monkeypatch.setattr(config, "CLEF_TIMEOUT_S", 2.5)
    clef_api.reply(_answers(clef_api))
    clef.decide(STATE, QUESTIONS)
    clef.decide(STATE, QUESTIONS, timeout=1.5)
    assert clef_api.requests[0]["timeout"] == {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}
    assert clef_api.requests[1]["timeout"]["read"] == 1.5


def test_decision_carries_typed_answers_tokens_and_request_id(clef_api):
    clef_api.reply(_answers(clef_api), input_tokens=287, request_id="req-42")
    decision = clef.decide(STATE, QUESTIONS)
    assert decision.answers["urgent"] == clef.NoulAnswer(0.83)
    assert decision.answers["team"] == clef.ChoiceAnswer(
        "technical", {"billing": 0.05, "technical": 0.9, "sales": 0.05})
    severity = decision.answers["severity"]
    assert severity.probabilities == {0: 0.05, 1: 0.15, 2: 0.6, 3: 0.2}
    assert severity.legend == {0: "No impact", 1: "Minor", 2: "Major", 3: "Critical"}
    assert severity.score == pytest.approx(1.95)
    assert (decision.input_tokens, decision.output_tokens, decision.request_id) == (287, 0, "req-42")
    assert decision.latency_ms >= 0


def test_request_id_is_optional(clef_api):
    clef_api.reply(_answers(clef_api), request_id=None, envelope=False)
    assert clef.decide(STATE, QUESTIONS).request_id is None


def test_a_call_without_the_fake_fails_loudly(live_clef_calls):
    with pytest.raises(AssertionError, match="live Clef call"):
        clef.decide(STATE, QUESTIONS)
    assert live_clef_calls == ["MainThread"]   # on record too, for a caller that swallows the error
    live_clef_calls.clear()   # made on purpose here; in any other test it fails at teardown


def test_a_call_made_after_its_test_ended_is_still_refused(monkeypatch, live_clef_calls):
    # A shadow check still queued when its test ends runs with the test's
    # monkeypatches undone (at the latest as the interpreter exits). It must
    # still be refused, carry no real token and write to no real ledger.
    monkeypatch.undo()
    hosts = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                        lambda transport, request: hosts.append(request.url.host) or httpx.Response(503))
    with pytest.raises((AssertionError, clef.ClefError)):
        clef.decide(STATE, QUESTIONS)
    # Compared outside the assert, so a failure can never print a real key.
    test_credentials = (config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN) == ("acct-test", "tok-test")
    assert hosts == [] and test_credentials
    backend = os.path.dirname(os.path.abspath(config.__file__))
    for path in (config.METERING_DB, config.GUARD_LOG_DB):
        assert not os.path.abspath(os.path.join(backend, path)).startswith(os.path.join(backend, "data") + os.sep)
    assert live_clef_calls == ["MainThread"]
    live_clef_calls.clear()


# --- Local limits ---------------------------------------------------------

def _nouls(n):
    return {f"q{i}": clef.Noul("Is it urgent?") for i in range(n)}


def _options(n):
    return {f"o{i}": f"Option {i}" for i in range(n)}


def _levels(n):
    return [f"Level {i}" for i in range(n)]


OUTSIDE_THE_LIMITS = {
    "no questions": ("clef", {}),
    "65 questions": ("clef", _nouls(65)),
    "id with a space": ("clef", {"is urgent": clef.Noul("Urgent?")}),
    "101-char id": ("clef", {"q" * 101: clef.Noul("Urgent?")}),
    "id ending in a newline": ("clef", {"urgent\n": clef.Noul("Urgent?")}),
    "1 choice option": ("clef", {"team": clef.Choice("Which team?", _options(1))}),
    "256 choice options": ("clef", {"team": clef.Choice("Which team?", _options(256))}),
    "1 score level": ("clef", {"severity": clef.Score("How severe?", _levels(1))}),
    "11 score levels": ("clef", {"severity": clef.Score("How severe?", _levels(11))}),
    "no instructions": ("clef", {"urgent": clef.Noul("")}),
    "not a question": ("clef", {"urgent": {"type": "noul", "instructions": "Urgent?"}}),
    "unknown model": ("clef-max", {"urgent": clef.Noul("Urgent?")}),
}


@pytest.mark.parametrize("model, questions", list(OUTSIDE_THE_LIMITS.values()), ids=list(OUTSIDE_THE_LIMITS))
def test_requests_outside_the_limits_fail_before_any_http_call(clef_api, model, questions):
    clef_api.reply({})
    _fails("bad_request", STATE, questions, model=model)
    assert clef_api.requests == []


def test_the_limits_are_inclusive():
    clef.build_body("clef", STATE, _nouls(1))
    clef.build_body("clef", STATE, _nouls(64))
    clef.build_body("clef", STATE, {"q" * 100: clef.Noul("Urgent?"), "a.b-c_D9": clef.Noul("Urgent?")})
    for n in (2, 255):
        clef.build_body("clef", STATE, {"team": clef.Choice("Which team?", _options(n))})
    for n in (2, 10):
        clef.build_body("clef", STATE, {"severity": clef.Score("How severe?", _levels(n))})


# --- Parsing --------------------------------------------------------------

def test_live_response_parses_into_typed_answers():
    answers, input_tokens, output_tokens = clef.parse(LIVE_RESPONSE, LIVE_QUESTIONS)
    assert answers["category"] == clef.ChoiceAnswer(
        "bug_report", {"bug_report": 0.9664, "feature_request": 0.0135, "billing": 0.0058, "other": 0.0143})
    assert answers["severity"] == clef.ScoreAnswer(
        1.1069, {0: 0.0636, 1: 0.766, 2: 0.1704}, dict(enumerate(LIVE_QUESTIONS["severity"].criteria)))
    assert answers["has_repro_steps"] == clef.NoulAnswer(0.5355)
    assert (input_tokens, output_tokens) == (395, 0)


def test_a_bare_result_is_accepted():
    assert clef.parse(LIVE_RESPONSE["result"], LIVE_QUESTIONS) == clef.parse(LIVE_RESPONSE, LIVE_QUESTIONS)


@pytest.mark.parametrize("code, kind", [(3036, "quota"), (3040, "capacity"), (5007, "protocol"), (None, "protocol")])
def test_a_failure_envelope_is_an_error(code, kind):
    payload = {"result": None, "success": False, "errors": [{"code": code, "message": "No such model"}],
               "messages": []}
    with pytest.raises(clef.ClefError) as err:
        clef.parse(payload, LIVE_QUESTIONS)
    assert (err.value.kind, err.value.code) == (kind, code)


def _broken(change):
    payload = copy.deepcopy(LIVE_RESPONSE)
    change(payload["result"])
    return payload


def _answer(result, qid):
    return result["answers"][qid]


MALFORMED = {
    "missing answer": lambda r: r["answers"].pop("has_repro_steps"),
    "wrong type": lambda r: _answer(r, "category").update(type="noul"),
    "choice not an option": lambda r: _answer(r, "category").update(choice="spam"),
    "probabilities sum to 0.97": lambda r: _answer(r, "category")["probabilities"].update(bug_report=0.9364),
    "probability for an unknown option": lambda r: _answer(r, "category")["probabilities"].update(spam=0.0),
    "an option without a probability": lambda r: _answer(r, "category")["probabilities"].pop("other"),
    "noul above 1": lambda r: _answer(r, "has_repro_steps").update(noul=1.5),
    "noul as text": lambda r: _answer(r, "has_repro_steps").update(noul="0.5"),
    "score level that was not asked": lambda r: _answer(r, "severity")["probabilities"].update({"3": 0.0}),
    "score level that is not a number": lambda r: _answer(r, "severity")["probabilities"].update({"x": 0.0}),
    "score without a legend": lambda r: _answer(r, "severity").pop("legend"),
    "no usage": lambda r: r.pop("usage"),
    "no answers": lambda r: r.pop("answers"),
}


@pytest.mark.parametrize("change", list(MALFORMED.values()), ids=list(MALFORMED))
def test_malformed_answers_are_protocol_errors(change):
    with pytest.raises(clef.ClefError) as err:
        clef.parse(_broken(change), LIVE_QUESTIONS)
    assert err.value.kind == "protocol"


@pytest.mark.parametrize("payload", [[], {"result": "oops", "success": True}], ids=["a list", "result not an object"])
def test_a_response_that_is_not_an_object_is_a_protocol_error(payload):
    with pytest.raises(clef.ClefError) as err:
        clef.parse(payload, LIVE_QUESTIONS)
    assert err.value.kind == "protocol"


def test_probabilities_may_be_off_by_rounding():
    payload = _broken(lambda r: _answer(r, "category")["probabilities"].update(bug_report=0.9564))  # sums to 0.99
    answers, _, _ = clef.parse(payload, LIVE_QUESTIONS)
    assert answers["category"].probabilities["bug_report"] == 0.9564


# --- Errors ---------------------------------------------------------------

HTTP_ERRORS = [
    (401, 10000, "auth"), (403, None, "auth"),
    (429, 3036, "quota"), (429, 3040, "capacity"),
    (400, 5006, "bad_request"), (404, 3042, "bad_request"), (413, 3006, "bad_request"), (422, None, "bad_request"),
    (408, 3007, "timeout"),
    (500, None, "server"), (503, None, "server"),
]


@pytest.mark.parametrize("status, code, kind", HTTP_ERRORS)
def test_http_errors_map_to_kinds(clef_api, status, code, kind):
    clef_api.error(status, code)
    error = _fails(kind, STATE, QUESTIONS)
    assert (error.status, error.code) == (status, code)


@pytest.mark.parametrize("exc, kind", [
    (httpx.ReadTimeout("timed out"), "timeout"),
    (httpx.ConnectTimeout("timed out"), "timeout"),
    (httpx.ConnectError("unreachable"), "server"),
], ids=["read timeout", "connect timeout", "unreachable"])
def test_transport_failures_map_to_kinds(clef_api, exc, kind):
    clef_api.raise_(exc)
    _fails(kind, STATE, QUESTIONS)


@pytest.mark.parametrize("status, kind", [(200, "protocol"), (502, "server")])
def test_a_body_that_is_not_json(clef_api, status, kind):
    clef_api.raw(status, "<html>Bad gateway</html>")
    _fails(kind, STATE, QUESTIONS)


def test_malformed_answers_from_the_api_are_protocol_errors(clef_api):
    clef_api.reply({"urgent": clef_api.noul(0.5)})  # the other two answers are missing
    _fails("protocol", STATE, QUESTIONS)


@pytest.mark.parametrize("missing", ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"])
def test_missing_credentials_fail_without_a_request(clef_api, monkeypatch, missing):
    clef_api.reply(_answers(clef_api))
    assert clef.is_configured()
    monkeypatch.setattr(config, missing, "")
    assert not clef.is_configured()
    _fails("config", STATE, QUESTIONS)
    assert clef_api.requests == []


@pytest.mark.parametrize("setting, value", [
    ("CLOUDFLARE_API_TOKEN", "tok-test​"),        # a zero-width space, copied from a web page
    ("CLOUDFLARE_API_TOKEN", "“tok-test”"),  # curly quotes
    ("CLOUDFLARE_API_TOKEN", "tok test"),
    ("CLOUDFLARE_ACCOUNT_ID", "acct-test/"),
], ids=["zero-width space", "curly quotes", "a space", "a slash in the account"])
def test_credentials_pasted_with_stray_characters_fail_without_a_request(clef_api, monkeypatch, setting, value):
    clef_api.reply(_answers(clef_api))
    monkeypatch.setattr(config, setting, value)
    error = _fails("config", STATE, QUESTIONS)
    assert clef_api.requests == [] and setting in str(error) and value not in str(error)


def _fresh_config(monkeypatch, **env):
    """config.py loaded as a new module from the given env only (backend/.env ignored)."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "CLEF_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("fresh_config", config.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clef_is_off_by_default_with_a_3_second_budget(monkeypatch):
    fresh = _fresh_config(monkeypatch)
    # Compared outside the assert, so a failure can never print a real key.
    unset = fresh.CLOUDFLARE_ACCOUNT_ID == "" and fresh.CLOUDFLARE_API_TOKEN == ""
    assert unset
    assert fresh.CLEF_TIMEOUT_S == 3


def test_clef_settings_come_from_the_environment(monkeypatch):
    fresh = _fresh_config(monkeypatch, CLOUDFLARE_ACCOUNT_ID="acct-env", CLOUDFLARE_API_TOKEN="tok-env",
                          CLEF_TIMEOUT_S="1.5")
    settings = (fresh.CLOUDFLARE_ACCOUNT_ID, fresh.CLOUDFLARE_API_TOKEN, fresh.CLEF_TIMEOUT_S)
    assert settings == ("acct-env", "tok-env", 1.5)


# --- Daily quota ----------------------------------------------------------

def test_a_used_up_quota_pauses_clef_until_midnight_utc(clef_api, monkeypatch, caplog):
    now = [datetime(2026, 10, 8, 13, 30, tzinfo=timezone.utc)]
    monkeypatch.setattr(clef, "_utcnow", lambda: now[0])
    clef_api.error(429, 3036, "You have used up your daily free allocation of 10,000 neurons.")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            _fails("quota", STATE, QUESTIONS)
    assert len(clef_api.requests) == 1
    assert clef.quota_exhausted_until() == datetime(2026, 10, 9, tzinfo=timezone.utc)
    (record,) = [r for r in caplog.records if r.name == "clef"]
    assert record.levelno == logging.ERROR
    assert "00:00 UTC" in record.getMessage() and "05:00 PKT" in record.getMessage()

    now[0] = datetime(2026, 10, 8, 23, 59, 59, tzinfo=timezone.utc)
    _fails("quota", STATE, QUESTIONS)
    assert len(clef_api.requests) == 1

    now[0] = datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc)
    assert clef.quota_exhausted_until() is None
    clef_api.reply(_answers(clef_api))
    clef.decide(STATE, QUESTIONS)
    assert len(clef_api.requests) == 2


def test_a_used_up_quota_just_after_midnight_pauses_clef_briefly(clef_api, monkeypatch):
    # Our clock may run ahead of Cloudflare's, or an answer sent before 00:00 may
    # arrive after it: the allowance then resets seconds later, not a day later.
    now = [datetime(2026, 10, 8, 23, 59, 58, tzinfo=timezone.utc)]
    monkeypatch.setattr(clef, "_utcnow", lambda: now[0])
    clef_api.error(429, 3036)
    _fails("quota", STATE, QUESTIONS)
    assert clef.quota_exhausted_until() == datetime(2026, 10, 9, tzinfo=timezone.utc)

    now[0] = datetime(2026, 10, 9, 0, 0, 1, tzinfo=timezone.utc)
    _fails("quota", STATE, QUESTIONS)   # not reset on Cloudflare's side yet
    assert len(clef_api.requests) == 2
    assert clef.quota_exhausted_until() <= datetime(2026, 10, 9, 0, 5, tzinfo=timezone.utc)
    _fails("quota", STATE, QUESTIONS)   # paused: no request
    assert len(clef_api.requests) == 2

    now[0] = datetime(2026, 10, 9, 0, 5, tzinfo=timezone.utc)
    clef_api.reply(_answers(clef_api))
    clef.decide(STATE, QUESTIONS)
    assert len(clef_api.requests) == 3

    # Still used up well after midnight: that is today's allowance gone.
    now[0] = datetime(2026, 10, 9, 0, 30, tzinfo=timezone.utc)
    clef_api.error(429, 3036)
    _fails("quota", STATE, QUESTIONS)
    assert clef.quota_exhausted_until() == datetime(2026, 10, 10, tzinfo=timezone.utc)


def test_parallel_calls_that_hit_the_quota_log_it_once(clef_api, caplog):
    clef_api.error(429, 3036)
    both_sent = threading.Barrier(2)
    clef_api.before = lambda body: both_sent.wait(timeout=5)
    with caplog.at_level(logging.WARNING), ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(clef.decide, STATE, QUESTIONS) for _ in range(2)]
        kinds = [f.exception().kind for f in futures]
    assert kinds == ["quota", "quota"] and len(clef_api.requests) == 2
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


def test_a_busy_model_does_not_pause_clef(clef_api):
    clef_api.error(429, 3040, "Capacity temporarily exceeded, please try again.")
    _fails("capacity", STATE, QUESTIONS)
    _fails("capacity", STATE, QUESTIONS)
    assert len(clef_api.requests) == 2 and clef.quota_exhausted_until() is None


# --- Logs -----------------------------------------------------------------

SENTINEL = "SENTINEL-4f2a call me on 0300-1234567"


def test_a_failed_call_is_logged_once_with_its_details(clef_api, caplog):
    clef_api.error(503, 1031)
    with caplog.at_level(logging.WARNING):
        _fails("server", STATE, QUESTIONS, model="clef-flash")
    (record,) = [r for r in caplog.records if r.name == "clef"]
    assert record.levelno == logging.WARNING
    for part in ("@cf/cloudflare/clef-flash", "server", "503", "1031", "req-test", " ms"):
        assert part in record.getMessage()


def test_logs_and_errors_never_contain_the_state_or_the_token(clef_api, caplog):
    failures = [
        lambda: clef_api.error(503),
        # An error message that echoes the input must not reach the logs either.
        lambda: clef_api.error(422, None, f"Extra inputs are not permitted, input_value='{SENTINEL}'"),
        lambda: clef_api.raise_(httpx.ReadTimeout("timed out")),
        lambda: clef_api.reply({"urgent": clef_api.noul(2.0)}),
        lambda: clef_api.raw(200, f"<html>{SENTINEL}</html>"),
        lambda: clef_api.error(429, 3036, f"used up: {SENTINEL}"),
    ]
    errors = []
    with caplog.at_level(logging.DEBUG):
        for fail in failures:
            fail()
            with pytest.raises(clef.ClefError) as err:
                clef.decide({"message": SENTINEL}, QUESTIONS)
            errors.append(err.value)
        with pytest.raises(clef.ClefError) as err:  # paused now: refused without a request
            clef.decide({"message": SENTINEL}, QUESTIONS)
        errors.append(err.value)
    assert [e.kind for e in errors] == ["server", "bad_request", "timeout", "protocol", "protocol", "quota", "quota"]
    text = caplog.text + " ".join(str(e) for e in errors)
    assert SENTINEL not in text and "tok-test" not in text
    # Not vacuous: each failed request was logged.
    assert len([r for r in caplog.records if r.name == "clef"]) == len(failures)


# --- Metering -------------------------------------------------------------

def test_a_call_writes_one_ledger_row(clef_api):
    clef_api.reply(_answers(clef_api))  # 346 input tokens
    with metering.context(client_id="acme", session_id="s1"):
        clef.decide(STATE, QUESTIONS, feature="guard")
        assert metering.current()["feature"] == "unknown"  # feature= covers this call only
    (row,) = _rows()
    assert (row["provider"], row["model"], row["feature"]) == ("cloudflare", "@cf/cloudflare/clef", "guard")
    assert (row["client_id"], row["session_id"]) == ("acme", metering.session_ref("s1"))
    assert (row["input_tokens"], row["output_tokens"], row["total_tokens"], row["estimated"]) == (346, 0, 346, 0)
    assert row["cost_usd"] == round(346 * 0.24 / 1e6, 8)


def test_the_feature_comes_from_the_context_when_not_given(clef_api):
    clef_api.reply(_answers(clef_api))
    with metering.feature("alfred.route"):
        clef.decide(STATE, QUESTIONS, model="clef-flash")
    (row,) = _rows()
    assert (row["feature"], row["model"]) == ("alfred.route", "@cf/cloudflare/clef-flash")
    assert row["cost_usd"] == round(346 * 0.09 / 1e6, 8)


def test_clef_prices_are_listed(monkeypatch):
    monkeypatch.setattr(metering, "_prices", None)  # read pricing.json afresh
    assert metering.estimate_cost("@cf/cloudflare/clef", 1_000_000, 0) == 0.24
    assert metering.estimate_cost("@cf/cloudflare/clef-flash", 1_000_000, 0) == 0.09


def test_failed_calls_write_no_ledger_row(clef_api, monkeypatch):
    for fail, kind in [
        (lambda: clef_api.error(401, 10000), "auth"),
        (lambda: clef_api.error(429, 3040), "capacity"),
        (lambda: clef_api.raise_(httpx.ReadTimeout("timed out")), "timeout"),
        (lambda: clef_api.reply({"urgent": clef_api.noul(0.5)}), "protocol"),
        (lambda: clef_api.raw(200, "not json"), "protocol"),
        (lambda: clef_api.error(429, 3036), "quota"),
        (lambda: None, "quota"),  # paused: no request at all
    ]:
        fail()
        _fails(kind, STATE, QUESTIONS, feature="guard")
    _fails("bad_request", STATE, {}, feature="guard")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "")
    _fails("config", STATE, QUESTIONS, feature="guard")
    assert _rows() == []


# --- Async ----------------------------------------------------------------

def test_adecide_works_like_decide(clef_api):
    clef_api.reply(_answers(clef_api))
    expected = clef.decide(STATE, QUESTIONS, feature="guard")
    for _ in range(2):  # a fresh event loop each time
        decision = asyncio.run(clef.adecide(STATE, QUESTIONS, feature="guard"))
        assert (decision.model, decision.answers, decision.input_tokens, decision.request_id) == (
            expected.model, expected.answers, expected.input_tokens, expected.request_id)
    sent = clef_api.requests
    assert len(sent) == 3 and all((r["url"], r["json"]) == (sent[0]["url"], sent[0]["json"]) for r in sent)
    assert [(r["provider"], r["feature"], r["input_tokens"]) for r in _rows()] == [("cloudflare", "guard", 346)] * 3


@pytest.mark.parametrize("fail, kind", [
    (lambda fake: fake.error(429, 3040), "capacity"),
    (lambda fake: fake.raise_(httpx.ReadTimeout("timed out")), "timeout"),
    (lambda fake: fake.reply({}), "protocol"),
], ids=["capacity", "timeout", "protocol"])
def test_adecide_errors_map_like_decide(clef_api, fail, kind):
    fail(clef_api)
    with pytest.raises(clef.ClefError) as err:
        asyncio.run(clef.adecide(STATE, QUESTIONS))
    assert err.value.kind == kind
    assert _rows() == []


def test_adecide_gives_up_at_its_timeout_however_slowly_the_answer_comes(monkeypatch):
    # httpx's timeout applies to each phase and each read; the call as a whole
    # must still end at the timeout.
    async def trickling(request):
        await asyncio.sleep(2)
        return httpx.Response(200, json={})
    monkeypatch.setattr(clef, "_transport", httpx.MockTransport(trickling))
    started = time.perf_counter()
    with pytest.raises(clef.ClefError) as err:
        asyncio.run(clef.adecide(STATE, QUESTIONS, timeout=0.2))
    assert err.value.kind == "timeout" and time.perf_counter() - started < 1


def test_adecide_refuses_locally_like_decide(clef_api, monkeypatch):
    clef_api.reply(_answers(clef_api))
    with pytest.raises(clef.ClefError) as err:
        asyncio.run(clef.adecide(STATE, {}))
    assert err.value.kind == "bad_request"
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "")
    with pytest.raises(clef.ClefError) as err:
        asyncio.run(clef.adecide(STATE, QUESTIONS))
    assert err.value.kind == "config"
    assert clef_api.requests == []
