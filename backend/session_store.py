"""In-memory session state for the Ava chat.

Each chat session keeps its uploaded CV (text + vector store), Ava's memory
thread id, and any documents she has generated for download.
"""

import uuid
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    kind: str          # "cv" | "cover_letter"
    title: str
    content: str       # the Markdown Ava wrote
    docx_path: str
    pdf_path: str


@dataclass
class Session:
    vector_store: object = None
    cv_text: str = ""
    has_cv: bool = False
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    documents: dict = field(default_factory=dict)  # id -> Document


_sessions: dict[str, Session] = {}


def create_session() -> tuple[str, Session]:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session()
    return session_id, _sessions[session_id]


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def find_document(doc_id: str) -> Document | None:
    for session in _sessions.values():
        doc = session.documents.get(doc_id)
        if doc:
            return doc
    return None
