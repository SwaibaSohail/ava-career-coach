import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import actions
import config
import session_store
from session_store import Session


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    # No live Groq from the ingress guard, and isolate the global session store.
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", False)
    monkeypatch.setattr(session_store, "_sessions", {})


def test_is_smtp_configured(monkeypatch):
    for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.setattr(config, name, "x")
    assert config.is_smtp_configured() is True
    monkeypatch.setattr(config, "SMTP_PASSWORD", "")
    assert config.is_smtp_configured() is False


def test_pending_action_store_roundtrip():
    from session_store import Session, PendingAction, find_pending_action
    s = Session()
    a = PendingAction(id="a1", kind="email", params={"to": "x@y.com"})
    s.pending_actions[a.id] = a
    assert find_pending_action(s, "a1") is a
    assert find_pending_action(s, "nope") is None
    assert a.status == "pending" and a.error is None


def test_build_email_action_stores_pending():
    import actions
    from session_store import Session
    s = Session()
    a = actions.build_email_action(s, "jobs@acme.com", "Application", "Hello there.")
    assert a.status == "pending" and a.kind == "email"
    assert s.pending_actions[a.id] is a
    assert a.params == {"to": "jobs@acme.com", "subject": "Application", "body": "Hello there."}


@pytest.mark.parametrize("to,subject,body", [
    ("not-an-email", "Hi", "Body"),
    ("jobs@acme.com", "Sub\r\nBcc: evil@x.com", "Body"),  # header injection
    ("jobs@acme.com", "Hi", "x" * 20001),                 # oversized body
])
def test_build_email_action_rejects_bad_input(to, subject, body):
    import actions
    from session_store import Session
    s = Session()
    with pytest.raises(ValueError):
        actions.build_email_action(s, to, subject, body)
    assert s.pending_actions == {}


def _pending(s):
    return actions.build_email_action(s, "jobs@acme.com", "Application", "Hello there.")


def test_execute_email_sends_once(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"] = host
            sent["port"] = port
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def starttls(self):
            sent["tls"] = True
        def login(self, u, p):
            sent["login"] = u
        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setattr(actions.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(config, "SMTP_HOST", "smtp.x")
    monkeypatch.setattr(config, "SMTP_PORT", 587)
    monkeypatch.setattr(config, "SMTP_USER", "me@x.com")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "pw")
    monkeypatch.setattr(config, "SMTP_FROM", "me@x.com")

    s = Session()
    a = _pending(s)
    actions.execute_email(a)
    assert a.status == "sent"
    assert sent["msg"]["To"] == "jobs@acme.com"
    assert sent["msg"]["From"] == "me@x.com"
    assert sent["msg"]["Subject"] == "Application"
    assert sent["tls"] is True and sent["login"] == "me@x.com"


def test_execute_email_no_resend_when_sent(monkeypatch):
    calls = []
    monkeypatch.setattr(actions, "send_email_smtp", lambda *a, **k: calls.append(1))
    s = Session()
    a = _pending(s)
    a.status = "sent"
    actions.execute_email(a)
    assert calls == []


def test_execute_email_retries_when_failed(monkeypatch):
    calls = []
    monkeypatch.setattr(actions, "send_email_smtp", lambda *a, **k: calls.append(1))
    s = Session()
    a = _pending(s)
    a.status = "failed"
    actions.execute_email(a)
    assert calls == [1] and a.status == "sent"


def test_execute_email_no_send_when_sending(monkeypatch):
    calls = []
    monkeypatch.setattr(actions, "send_email_smtp", lambda *a, **k: calls.append(1))
    s = Session()
    a = _pending(s)
    a.status = "sending"
    actions.execute_email(a)
    assert calls == []


def test_execute_email_failure_sets_failed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(actions, "send_email_smtp", boom)
    s = Session()
    a = _pending(s)
    actions.execute_email(a)
    assert a.status == "failed" and "smtp down" in (a.error or "")


def test_propose_email_tool_creates_pending_without_sending(monkeypatch):
    calls = []
    monkeypatch.setattr(actions, "send_email_smtp", lambda *a, **k: calls.append(1))
    from agent import _session_tools
    s = Session()
    tools = {t.name: t for t in _session_tools(s)}
    out = tools["propose_email"].invoke({"to": "jobs@acme.com", "subject": "Hi", "body": "Hello."})
    assert len(s.pending_actions) == 1
    assert "approve" in out.lower() and calls == []


def test_propose_email_tool_reports_bad_recipient():
    from agent import _session_tools
    s = Session()
    tools = {t.name: t for t in _session_tools(s)}
    out = tools["propose_email"].invoke({"to": "nope", "subject": "Hi", "body": "Hello."})
    assert s.pending_actions == {}
    assert "valid" in out.lower() or "address" in out.lower()
