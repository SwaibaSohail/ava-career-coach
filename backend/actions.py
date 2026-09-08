"""Outward actions with a human-approval gate. Slice 1: send an email.

`build_email_action` validates + stores a PendingAction (it never sends).
`execute_email` performs the SMTP send, guarded so one action can't send twice.
"""

import re
import smtplib
import threading
import uuid
from email.message import EmailMessage

import config
from session_store import PendingAction

_MAX_BODY = 20000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_send_lock = threading.Lock()


def build_email_action(session, to: str, subject: str, body: str) -> PendingAction:
    """Validate and store a pending email. Raises ValueError on bad input."""
    to = (to or "").strip()
    subject = (subject or "").strip()
    body = body or ""
    # Header-injection guard first, so a CRLF recipient gets the header message.
    if any(c in to + subject for c in ("\r", "\n")):
        raise ValueError("Email recipient and subject may not contain line breaks.")
    if not _EMAIL_RE.match(to):
        raise ValueError(f"'{to}' is not a valid email address.")
    if len(body) > _MAX_BODY:
        raise ValueError(f"The email body is too long (max {_MAX_BODY} characters).")
    action = PendingAction(
        id=uuid.uuid4().hex,
        kind="email",
        params={"to": to, "subject": subject, "body": body},
    )
    session.pending_actions[action.id] = action
    return action


def send_email_smtp(to: str, subject: str, body: str) -> None:
    """Send one plain-text email via SMTP (587 + STARTTLS). Raises on failure."""
    msg = EmailMessage()
    msg["From"] = config.SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
        smtp.send_message(msg)


def execute_email(action: PendingAction) -> PendingAction:
    """Send a pending/failed email exactly once."""
    # The lock guards only the status check-and-set (microseconds), NOT the send,
    # so throughput is unaffected and a per-action lock (which a second concurrent
    # confirm couldn't see) is unnecessary.
    with _send_lock:
        if action.status not in ("pending", "failed"):
            return action
        action.status = "sending"
        action.error = None
    try:
        send_email_smtp(action.params["to"], action.params["subject"], action.params["body"])
        action.status = "sent"
    except Exception as exc:
        action.status = "failed"
        action.error = str(exc)
    return action
