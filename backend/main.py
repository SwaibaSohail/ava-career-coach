"""FastAPI backend for the Ava career-coach chat.

Endpoints are thin wrappers around the conversational agent: start a session,
attach a CV, exchange messages, and download the documents Ava generates.
CORS is enabled for the React dev server on port 5173.
"""

import json
import os
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import config
from cv_processor import load_and_chunk_cv, extract_full_text
from vector_store import build_cv_vector_store
from agent import stream_ava, AVA_GREETING
from guardrails import guard_incoming
from schemas import MessageRequest, SessionResponse, UploadResponse
from session_store import create_session, find_document, get_session

app = FastAPI(title="Ava — CV & Job Coach API")

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
    return {"groq": config.is_api_key_configured(), "tavily": config.is_tavily_configured()}


@app.post("/api/session", response_model=SessionResponse)
def start_session():
    """Begin a chat and return Ava's opening message."""
    session_id, _ = create_session()
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


def _process_cv(session, pdf_path: str) -> None:
    """Blocking CV parse + embed. Run off the event loop via run_in_threadpool."""
    session.vector_store = build_cv_vector_store(load_and_chunk_cv(pdf_path))
    session.cv_text = extract_full_text(pdf_path)
    session.has_cv = True
    # Fresh memory so earlier draft/example CVs can't leak into tailoring.
    session.thread_id = str(uuid.uuid4())


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
        await run_in_threadpool(_process_cv, session, pdf_path)
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


async def _message_events(session, user_message: str):
    """Yield SSE frames for one turn. Runs the ingress guardrails first; a
    blocked message streams a canned reply and never reaches the agent."""
    # Runs the (possibly LLM-backed) guard off the event loop so it can't block
    # other requests.
    guard = await run_in_threadpool(guard_incoming, user_message)
    if not guard.allowed:
        yield f"data: {json.dumps({'type': 'token', 'text': guard.safe_reply})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    async for kind, data in stream_ava(session, guard.cleaned_message):
        payload = {"type": kind}
        if kind == "token":
            payload["text"] = data
        elif kind == "document":
            payload.update(data)
        yield f"data: {json.dumps(payload)}\n\n"


@app.post("/api/message")
async def message(req: MessageRequest):
    """One conversation turn with Ava, streamed as Server-Sent Events.

    Each line is `data: {json}` where json.type is 'token' (a text delta),
    'document' (a saved file: id/kind/title), or 'done'. Incoming messages pass
    the Layer-1 ingress guardrails before reaching the agent.
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
