import asyncio
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


def _register_doc(session, tmp_path, doc_id="d1", title="Frontend Developer CV", kind="cv"):
    """Add a fake generated document with a real PDF file on disk."""
    from session_store import Document
    p = tmp_path / f"{doc_id}.pdf"
    p.write_bytes(b"%PDF-1.4 fake pdf bytes")
    session.documents[doc_id] = Document(
        id=doc_id, kind=kind, title=title, content="...",
        docx_path=str(tmp_path / f"{doc_id}.docx"), pdf_path=str(p),
    )
    return str(p)


def test_build_email_action_attaches_cv_by_kind(tmp_path):
    from session_store import Session
    s = Session()
    path = _register_doc(s, tmp_path)
    # Attach by KIND — the reliable path the model uses (no id threading).
    a = actions.build_email_action(s, "jobs@acme.com", "Application", "Hi.", attach="cv")
    # The plain fields are unchanged; the path stays out of params.
    assert a.params == {"to": "jobs@acme.com", "subject": "Application", "body": "Hi."}
    assert len(a.attachments) == 1
    att = a.attachments[0]
    assert att["path"] == path
    assert att["filename"] == "Frontend_Developer_CV.pdf"
    assert att["kind"] == "cv"


def test_build_email_action_attaches_by_exact_id(tmp_path):
    from session_store import Session
    s = Session()
    _register_doc(s, tmp_path)
    a = actions.build_email_action(s, "jobs@acme.com", "Application", "Hi.", attach="d1")
    assert a.attachments[0]["kind"] == "cv"


def test_build_email_action_attaches_latest_of_kind(tmp_path):
    from session_store import Session
    s = Session()
    _register_doc(s, tmp_path, doc_id="d1", title="Old CV")
    _register_doc(s, tmp_path, doc_id="d2", title="New CV")
    a = actions.build_email_action(s, "jobs@acme.com", "Hi", "Body", attach="cv")
    assert a.attachments[0]["filename"] == "New_CV.pdf"


def test_build_email_action_no_saved_doc_raises():
    from session_store import Session
    s = Session()
    with pytest.raises(ValueError):
        actions.build_email_action(s, "jobs@acme.com", "Hi", "Body", attach="cv")
    assert s.pending_actions == {}


def test_build_email_message_includes_pdf_attachment(tmp_path):
    from session_store import Session
    s = Session()
    _register_doc(s, tmp_path)
    a = actions.build_email_action(s, "jobs@acme.com", "Hi", "Body", attach="cv")
    msg = actions.build_email_message(a.params["to"], a.params["subject"], a.params["body"], a.attachments)
    atts = list(msg.iter_attachments())
    assert len(atts) == 1
    assert atts[0].get_filename() == "Frontend_Developer_CV.pdf"
    assert atts[0].get_content_type() == "application/pdf"


def test_execute_email_passes_attachments(monkeypatch, tmp_path):
    captured = {}
    def fake_send(to, subject, body, attachments=None):
        captured["attachments"] = attachments
    monkeypatch.setattr(actions, "send_email_smtp", fake_send)
    from session_store import Session
    s = Session()
    _register_doc(s, tmp_path)
    a = actions.build_email_action(s, "jobs@acme.com", "Hi", "Body", attach="cv")
    actions.execute_email(a)
    assert a.status == "sent"
    assert captured["attachments"] and captured["attachments"][0]["filename"].endswith(".pdf")


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


def test_propose_email_tool_attaches_document(tmp_path):
    from agent import _session_tools
    s = Session()
    _register_doc(s, tmp_path)
    tools = {t.name: t for t in _session_tools(s)}
    out = tools["propose_email"].invoke(
        {"to": "jobs@acme.com", "subject": "Hi", "body": "Hello.", "attach": "cv"}
    )
    action = next(iter(s.pending_actions.values()))
    assert len(action.attachments) == 1
    assert action.attachments[0]["filename"] == "Frontend_Developer_CV.pdf"
    assert "Frontend_Developer_CV.pdf" in out


def _collect(agen):
    async def run():
        return [x async for x in agen]
    return asyncio.run(run())


def test_message_events_emits_action_and_no_smtp(monkeypatch):
    import main
    import agent
    from langchain_core.messages import AIMessageChunk
    from session_store import PendingAction

    def fake_build(session):
        class FakeAgent:
            async def astream(self, *a, **k):
                session.pending_actions["e1"] = PendingAction(
                    id="e1", kind="email",
                    params={"to": "jobs@acme.com", "subject": "Hi", "body": "Hello."})
                yield (AIMessageChunk(content="Drafted."), {})
        return FakeAgent()
    monkeypatch.setattr(agent, "_build_ava", fake_build)
    sent = []
    monkeypatch.setattr(actions, "send_email_smtp", lambda *a, **k: sent.append(1))

    frames = "".join(_collect(main._message_events(Session(), "email jobs@acme.com")))
    assert '"type": "action"' in frames and "jobs@acme.com" in frames
    assert '"type": "done"' in frames
    assert sent == []


def test_action_event_exposes_filename_not_path(monkeypatch):
    import main
    import agent
    from langchain_core.messages import AIMessageChunk
    from session_store import PendingAction

    secret_path = "C:/secret/server/path/cv.pdf"

    def fake_build(session):
        class FakeAgent:
            async def astream(self, *a, **k):
                session.pending_actions["e1"] = PendingAction(
                    id="e1", kind="email",
                    params={"to": "jobs@acme.com", "subject": "Hi", "body": "Hello."},
                    attachments=[{"path": secret_path, "filename": "My_CV.pdf", "kind": "cv"}],
                )
                yield (AIMessageChunk(content="Drafted."), {})
        return FakeAgent()
    monkeypatch.setattr(agent, "_build_ava", fake_build)

    frames = "".join(_collect(main._message_events(Session(), "email my cv")))
    assert "My_CV.pdf" in frames        # filename is shown to the client
    assert secret_path not in frames     # server path is never sent


def test_confirm_endpoint_sends(monkeypatch):
    import main
    from session_store import create_session
    monkeypatch.setattr(main, "is_smtp_configured", lambda: True)
    calls = []

    def fake_exec(a):
        a.status = "sent"
        calls.append(1)
        return a
    monkeypatch.setattr(main, "execute_email", fake_exec)
    sid, session = create_session()
    a = actions.build_email_action(session, "jobs@acme.com", "Hi", "Hello.")
    res = asyncio.run(main.confirm_action(main.ActionRequest(session_id=sid, action_id=a.id)))
    assert res.status == "sent" and calls == [1]


def test_confirm_endpoint_reports_unconfigured(monkeypatch):
    import main
    from session_store import create_session
    monkeypatch.setattr(main, "is_smtp_configured", lambda: False)
    sid, session = create_session()
    a = actions.build_email_action(session, "jobs@acme.com", "Hi", "Hello.")
    res = asyncio.run(main.confirm_action(main.ActionRequest(session_id=sid, action_id=a.id)))
    assert res.status == "pending" and res.error


def test_cancel_endpoint_cancels_pending():
    import main
    from session_store import create_session
    sid, session = create_session()
    a = actions.build_email_action(session, "jobs@acme.com", "Hi", "Hello.")
    res = asyncio.run(main.cancel_action(main.ActionRequest(session_id=sid, action_id=a.id)))
    assert res.status == "cancelled"


def test_cancel_does_not_unsend():
    import main
    from session_store import create_session
    sid, session = create_session()
    a = actions.build_email_action(session, "jobs@acme.com", "Hi", "Hello.")
    a.status = "sent"
    res = asyncio.run(main.cancel_action(main.ActionRequest(session_id=sid, action_id=a.id)))
    assert res.status == "sent"
