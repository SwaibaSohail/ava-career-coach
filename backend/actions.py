"""Outward actions with a human-approval gate. Slice 1: send an email.

`build_email_action` validates + stores a PendingAction (it never sends).
`execute_email` performs the SMTP send, guarded so one action can't send twice.
"""

import os
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

_PDF_MEDIA = ("application", "pdf")
_DOCX_MEDIA = ("application", "vnd.openxmlformats-officedocument.wordprocessingml.document")


def _safe_filename(title: str, kind: str, ext: str) -> str:
    """A clean download-style filename, e.g. 'Frontend_Developer_CV.pdf'."""
    base = (title or kind or "document").strip().replace(" ", "_")
    base = re.sub(r"[^A-Za-z0-9_.\-]", "", base) or "document"
    return base + ext


def _latest_document(session, kind: str):
    """Most recently saved session document of a kind ('cv' | 'cover_letter')."""
    for doc in reversed(list(session.documents.values())):
        if doc.kind == kind:
            return doc
    return None


def _document_attachment(doc) -> dict:
    """A safe attachment for a document Ava generated (read from disk at send time)."""
    if not doc.pdf_path or not os.path.exists(doc.pdf_path):
        raise ValueError("That document's file isn't available to attach. Try regenerating it.")
    return {
        "path": doc.pdf_path,
        "filename": _safe_filename(doc.title, doc.kind, ".pdf"),
        "kind": doc.kind,
    }


def _uploaded_cv_attachment(session):
    """A safe attachment for the CV the user uploaded (kept in memory, not on disk)."""
    data = getattr(session, "cv_pdf", None)
    if not data:
        return None
    raw = getattr(session, "cv_filename", "") or "CV.pdf"
    base = raw[:-4] if raw.lower().endswith(".pdf") else raw
    return {"data": data, "filename": _safe_filename(base, "cv", ".pdf"), "kind": "cv"}


def _resolve_attachment(session, ref: str) -> dict:
    """Turn an attachment reference into a safe attachment dict.

    `ref` is a document kind ('cv' / 'cover_letter' — the reliable path the model
    uses), the special value 'uploaded' for the CV the user uploaded, or an exact
    session document id. For 'cv', a CV Ava generated is preferred, falling back
    to the uploaded original. Only session documents or the uploaded CV can be
    attached — the source is read from server-side state, never from a path the
    model or client supplies, so an arbitrary filesystem path can't be smuggled in.
    """
    doc = session.documents.get(ref)  # exact id?
    if doc is not None:
        return _document_attachment(doc)

    key = (ref or "").strip().lower()
    if key in ("uploaded", "uploaded_cv", "original", "original_cv"):
        att = _uploaded_cv_attachment(session)
        if att is None:
            raise ValueError("There's no uploaded CV to attach — upload one first, then attach.")
        return att

    if "cover" in key:
        kind = "cover_letter"
    elif key in ("cv", "resume") or "cv" in key or "resume" in key:
        kind = "cv"
    else:
        raise ValueError(
            "I'm not sure which document to attach — save or upload the CV first, then attach."
        )

    doc = _latest_document(session, kind)
    if doc is not None:
        return _document_attachment(doc)
    # No generated CV — fall back to the uploaded original.
    if kind == "cv":
        att = _uploaded_cv_attachment(session)
        if att is not None:
            return att
    nice = kind.replace("_", " ")
    raise ValueError(f"There's no {nice} to attach yet — create or upload it first, then attach.")


def build_email_action(
    session, to: str, subject: str, body: str, attach: str = ""
) -> PendingAction:
    """Validate and store a pending email. Raises ValueError on bad input.

    If `attach` is given (a document kind like 'cv'/'cover_letter', or an exact
    document id), the matching session document is attached as a PDF; if none
    matches, ValueError is raised and nothing is stored.
    """
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
    attachments = []
    if attach:
        attachments.append(_resolve_attachment(session, attach))
    action = PendingAction(
        id=uuid.uuid4().hex,
        kind="email",
        params={"to": to, "subject": subject, "body": body},
        attachments=attachments,
    )
    session.pending_actions[action.id] = action
    return action


def _media_for(filename: str):
    return _DOCX_MEDIA if filename.lower().endswith(".docx") else _PDF_MEDIA


def build_email_message(to: str, subject: str, body: str, attachments=None) -> EmailMessage:
    """Assemble a plain-text email with optional file attachments."""
    msg = EmailMessage()
    msg["From"] = config.SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    for att in attachments or []:
        if att.get("data") is not None:
            data = att["data"]           # uploaded CV: bytes kept in memory
        else:
            with open(att["path"], "rb") as fh:
                data = fh.read()         # generated document: read from disk
        maintype, subtype = _media_for(att["filename"])
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=att["filename"])
    return msg


def send_email_smtp(to: str, subject: str, body: str, attachments=None) -> None:
    """Send one email (with any attachments) via SMTP (587 + STARTTLS). Raises on failure."""
    msg = build_email_message(to, subject, body, attachments)
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
        send_email_smtp(
            action.params["to"],
            action.params["subject"],
            action.params["body"],
            action.attachments,
        )
        action.status = "sent"
    except Exception as exc:
        action.status = "failed"
        action.error = str(exc)
    return action
