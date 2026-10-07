"""FastAPI backend for the Ava career-coach chat.

Endpoints are thin wrappers around the conversational agent: start a session,
attach a CV, exchange messages, and download the documents Ava generates.
Every model call is metered against the chat's client account (see metering.py);
GET /api/admin/usage is the internal billing report. CORS is enabled for the
React dev server on port 5173.
"""

import json
import logging
import os
import secrets
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import config
import metering
from actions import execute_email
from config import is_smtp_configured
from cv_processor import load_and_chunk_cv, extract_full_text
from vector_store import build_cv_vector_store
from agent import stream_ava, AVA_GREETING
from guardrails import guard_incoming
from interview import handle_turn
from mcp_client import init_mcp
from schemas import (
    ActionRequest,
    ActionResponse,
    MessageRequest,
    SessionResponse,
    UploadResponse,
)
from session_store import create_session, find_document, find_pending_action, get_session

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app):
    # Load MCP tools once at startup (fail-safe: never blocks the app on error).
    await init_mcp()
    yield


app = FastAPI(title="Ava — CV & Job Coach API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _require(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session. Start a new chat.")
    return session


@app.get("/")
def root():
    return {"status": "ok", "service": "Ava — CV & Job Coach API", "docs": "/docs"}


@app.get("/api/config")
def read_config():
    return {
        "groq": config.is_api_key_configured(),
        "tavily": config.is_tavily_configured(),
        "smtp": is_smtp_configured(),
    }


def _client_for_request(key: str | None) -> str:
    """The client account a new chat bills to, from its X-Client-Key header."""
    if not key:
        if config.REQUIRE_CLIENT_KEY:
            raise HTTPException(status_code=401, detail="A client key is required.")
        return metering.DEFAULT_CLIENT
    try:
        client_id = metering.client_for_key(key)
    except Exception:
        # Fail open: never block chats on a metering outage.
        log.error("metering: client key lookup failed; using the default client", exc_info=True)
        return metering.DEFAULT_CLIENT
    if client_id is None:
        raise HTTPException(status_code=401, detail="Unknown client key.")
    return client_id


@app.post("/api/session", response_model=SessionResponse)
def start_session(x_client_key: str | None = Header(default=None)):
    """Begin a chat and return Ava's opening message.

    An optional X-Client-Key header ties the chat to a client account for usage
    metering. The key identifies the tenant (it may ship in a browser bundle);
    it does not authenticate end users.
    """
    session_id, _ = create_session(client_id=_client_for_request(x_client_key))
    return SessionResponse(session_id=session_id, greeting=AVA_GREETING)


def _ensure_pdf(content: bytes) -> None:
    """Reject empty, oversized, or non-PDF uploads with a friendly 400."""
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"That file is too large (max {mb} MB).")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="That doesn't look like a PDF. Please upload a PDF CV.")


def _process_cv(session, pdf_path: str, content: bytes, filename: str) -> None:
    """Blocking CV parse + embed. Run off the event loop via run_in_threadpool."""
    # Build the new store first so a parse failure leaves the old CV intact.
    new_store = build_cv_vector_store(load_and_chunk_cv(pdf_path))
    cv_text = extract_full_text(pdf_path)
    old_store = session.vector_store
    session.vector_store = new_store
    session.cv_text = cv_text
    session.has_cv = True
    # Keep the original PDF bytes in memory so Ava can attach the uploaded CV.
    session.cv_pdf = content
    session.cv_filename = filename or "cv.pdf"
    # Fresh memory so earlier draft/example CVs can't leak into tailoring.
    session.thread_id = str(uuid.uuid4())
    # Drop the previous upload's collection so it doesn't linger in memory.
    if old_store is not None:
        try:
            old_store.delete_collection()
        except Exception:
            pass


@app.post("/api/upload", response_model=UploadResponse)
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    """Attach a CV PDF: validate it, chunk + embed it, and remember its text."""
    session = _require(session_id)
    content = await file.read()
    _ensure_pdf(content)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        pdf_path = tmp.name
    try:
        await run_in_threadpool(_process_cv, session, pdf_path, content, file.filename or "cv.pdf")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Couldn't read that PDF — it may be corrupted, encrypted, or password-protected.",
        )
    finally:
        os.unlink(pdf_path)
    return UploadResponse(ok=True, filename=file.filename or "cv.pdf", chars=len(session.cv_text))


_INTERVIEW_STILL_RUNNING = (
    "\n\nYour mock interview is still running — answer the current question, or "
    "type 'skip' or 'end interview'."
)

_LIMIT_REACHED = (
    "Your organisation has used this month's Ava allowance, so I can't reply right "
    "now. Please contact your administrator to raise the limit."
)

_LIMIT_BUSY = (
    "Your organisation is close to this month's Ava allowance and other chats are "
    "using the rest right now. Please try again in a moment."
)


async def _message_events(session, user_message: str):
    """Yield SSE frames for one turn. Checks the client's monthly token allowance
    first, then runs the ingress guardrails; a blocked message streams a canned
    reply and never reaches the agent. Every model call in the turn is metered
    under the session's client (the guard call is billed even when it blocks)."""
    with metering.context(client_id=session.client_id, session_id=session.id or None, feature="chat"):
        # Once per turn, before any model call. A turn that has started may
        # finish, so a client can go slightly over its allowance. The turn holds
        # a share of what's left until it ends, so parallel turns can't all start.
        allowance = await run_in_threadpool(metering.check_allowance, session.client_id, reserve=True)
        if not allowance.allowed:
            notice = _LIMIT_BUSY if allowance.used < allowance.limit else _LIMIT_REACHED
            yield f"data: {json.dumps({'type': 'token', 'text': notice})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        try:
            # Runs the (possibly LLM-backed) guard off the event loop so it can't block
            # other requests.
            guard = await run_in_threadpool(guard_incoming, user_message)
            interview = session.interview
            interview_active = interview is not None and interview.status == "active"
            if not guard.allowed:
                reply = guard.safe_reply
                if interview_active:
                    # The canned replies talk about CVs; make clear the interview hasn't moved on.
                    reply += _INTERVIEW_STILL_RUNNING
                yield f"data: {json.dumps({'type': 'token', 'text': reply})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            if interview_active:
                # Interview answers bypass the agent: a small focused evaluator handles them.
                for kind, data in await run_in_threadpool(handle_turn, session, guard.cleaned_message):
                    payload = {"type": kind}
                    if kind == "token":
                        payload["text"] = data
                    else:
                        payload.update(data)
                    yield f"data: {json.dumps(payload)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            async for kind, data in stream_ava(session, guard.cleaned_message):
                payload = {"type": kind}
                if kind == "token":
                    payload["text"] = data
                elif kind in ("document", "action", "interview"):
                    payload.update(data)
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            # Also on errors and when the user leaves mid-reply.
            metering.release(session.client_id, allowance)


@app.post("/api/message")
async def message(req: MessageRequest):
    """One conversation turn with Ava, streamed as Server-Sent Events.

    Each line is `data: {json}` where json.type is one of:
      - 'token'     a text delta (`text`)
      - 'document'  a saved file: id, kind, title
      - 'action'    an email awaiting approval: id, kind, to, subject, body, attachments
      - 'interview' the current mock-interview question: id, role, index, total,
                    question, kind
      - 'report'    the final interview report: role, readiness, band, answered,
                    skipped, total, strengths, improvements, per_question
      - 'done'      end of the turn
    Each turn first checks the client's monthly token allowance; over the limit,
    a short notice is streamed and no model is called. Incoming messages then
    pass the Layer-1 ingress guardrails. While an interview is active, answers
    go to the interview engine instead of the agent.
    """
    session = _require(req.session_id)
    return StreamingResponse(
        _message_events(session, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/document/{doc_id}")
def download(doc_id: str, fmt: str = "docx"):
    """Download a generated document as .docx (default) or .pdf via ?fmt=pdf."""
    doc = find_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if fmt == "pdf":
        path, media, ext = doc.pdf_path, "application/pdf", ".pdf"
    else:
        path, media, ext = doc.docx_path, _DOCX_MEDIA, ".docx"
    filename = (doc.title or doc.kind).replace(" ", "_") + ext
    return FileResponse(path, filename=filename, media_type=media)


@app.post("/api/action/confirm", response_model=ActionResponse)
async def confirm_action(req: ActionRequest):
    """Run an approved action (send the email). SMTP happens only here."""
    session = _require(req.session_id)
    action = find_pending_action(session, req.action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found.")
    if not is_smtp_configured():
        return ActionResponse(
            status=action.status,
            error="Email isn't set up — add SMTP settings to .env.",
        )
    result = await run_in_threadpool(execute_email, action)
    return ActionResponse(status=result.status, error=result.error)


@app.post("/api/action/cancel", response_model=ActionResponse)
async def cancel_action(req: ActionRequest):
    """Discard a pending/failed action so it can't be sent."""
    session = _require(req.session_id)
    action = find_pending_action(session, req.action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Action not found.")
    if action.status in ("pending", "failed"):
        action.status = "cancelled"
    return ActionResponse(status=action.status, error=action.error)


# Internal only: kept out of the public /docs page.
@app.get("/api/admin/usage", include_in_schema=False)
def admin_usage(month: str | None = None, client_id: str | None = None,
                x_admin_key: str | None = Header(default=None)):
    """Internal token-usage report for billing (UTC month; ?month=YYYY-MM).

    Disabled (404) unless ADMIN_API_KEY is set; requires a matching X-Admin-Key.
    """
    if not config.ADMIN_API_KEY:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_admin_key or not secrets.compare_digest(
        x_admin_key.encode("utf-8"), config.ADMIN_API_KEY.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid admin key.")
    try:
        return metering.usage_report(month=month, client_id=client_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
