"""Outward actions with a human-approval gate. Slice 1: send an email.

`build_email_action` validates + stores a PendingAction (it never sends).
`execute_email` performs the SMTP send, guarded so one action can't send twice.
"""

import re
import uuid

from session_store import PendingAction

_MAX_BODY = 20000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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
