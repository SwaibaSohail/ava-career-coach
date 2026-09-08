import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import config
import session_store


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
