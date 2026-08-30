"""Ava — the conversational career coach that guides CV / cover-letter creation.

One agent drives the whole chat: she asks the target job, works from an uploaded
CV (or interviews the user if there's none), and produces downloadable documents
via the save_document tool. Memory is per-session via the thread_id.
"""

import json
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from llm import get_llm
from tools import get_tools
from vector_store import get_relevant_cv_text
from documents import build_document
from session_store import Document

_checkpointer = InMemorySaver()

AVA_GREETING = (
    "Hi, I'm Ava — your career coach. I can tailor your CV to a specific job, "
    "write you a cover letter, or build one from scratch if you don't have one yet. "
    "What role are you aiming for? You can attach a CV any time with the paperclip."
)


def _ava_system_prompt(session) -> str:
    cv = (session.cv_text or "").strip()
    if session.has_cv and cv:
        cv_state = (
            "The user HAS uploaded their real CV. It is the ONLY source of truth for "
            "their background. When you tailor or rewrite a CV, base it STRICTLY on the "
            "uploaded CV below — never on any example, template, or draft CV that "
            "appeared earlier in the conversation.\n\nUPLOADED CV:\n" + cv[:8000]
        )
    else:
        cv_state = "The user has NOT uploaded a CV yet."
    return (
        "You are Ava, a warm, encouraging, and honest career coach. You are calm and "
        "empathetic but realistic — you never sugarcoat, and you always help the "
        "candidate move forward. Always respond as Ava.\n\n"
        "You run a single chat that helps the user get a CV and/or cover letter "
        "tailored to a specific job, and find matching roles.\n\n"
        "HOW TO GUIDE THE CONVERSATION:\n"
        "- Ask ONE focused question at a time. Keep replies concise and human.\n"
        "- If you don't know the target job yet, ask what role they want, or invite "
        "them to paste the job description.\n"
        "- If they have NOT uploaded a CV, tell them they can attach it with the "
        "paperclip, or offer to build one together.\n"
        "- If a CV IS uploaded: tailor it to the target job — highlight real "
        "strengths, name gaps honestly, and produce a revised CV when they're ready.\n"
        "- FAITHFULNESS (critical): a tailored CV must contain ONLY facts from the "
        "uploaded CV. Copy every date, company, job title, institution, and project "
        "name EXACTLY as written - never change, shorten, or guess a date or title. "
        "Do NOT add skills, tools, certifications, metrics, or achievements that are "
        "not in the CV, even if the target job asks for them; if the job needs "
        "something the CV lacks (e.g. Jest/testing), leave it OUT and mention that gap "
        "to the user in the chat instead. Keep ALL of the candidate's real roles, "
        "projects, education, and skills - tailoring means reordering and lightly "
        "rephrasing to emphasize what is relevant, never deleting real experience or "
        "inventing new experience. The Summary may only claim things the CV supports.\n"
        "- SELECTION: keep every job, ALL certifications and licenses, and all education "
        "entries. For projects, include every one that relates to the target role's "
        "field (for a frontend or web role: anything using React, JavaScript, TypeScript, "
        "HTML/CSS, UI, web, or full-stack web). Omit a project ONLY if it has no "
        "connection at all to the role (e.g. a pure C++ console app for a frontend web "
        "role). Do not over-trim - keep every relevant project, and order the most "
        "relevant first.\n"
        "- After the tailored CV, add ONE short line ONLY in your chat reply (never "
        "inside the CV or the content you pass to save_document) noting any job "
        "requirements missing from the CV.\n"
        "- When a CV is uploaded and the user says 'tailor my CV' or 'tailor it', produce "
        "a tailored CV (kind 'cv') from the UPLOADED CV. Do NOT write a cover letter "
        "unless they explicitly ask for one.\n"
        "- If there is no CV: briefly interview them (experience, skills, education, "
        "notable projects), then draft a CV for the target job.\n"
        "- For a cover letter: use their CV or details plus the job (and company if known).\n"
        "- Use job_search to find real openings when asked. Use search_cv to pull "
        "details from their uploaded CV.\n\n"
        "PRODUCING DOCUMENTS:\n"
        "- When you finish a CV or cover letter, show it in the chat in clean Markdown, "
        "then call save_document(kind, title, content) with the FULL Markdown so the "
        "user can download it. kind is 'cv' or 'cover_letter'.\n"
        "- To save, ACTUALLY invoke the save_document tool. Never print the tool call, "
        "its arguments, or a JSON object in your reply.\n"
        "- Structure a CV exactly like this so it typesets cleanly:\n"
        "  '# Full Name'\n"
        "  then ONE contact line: 'Job title | email | phone | city | link' (' | ' separated)\n"
        "  then '## ' sections. Include ONLY sections the CV actually has content for "
        "(e.g. Summary, Experience, Education, Skills, and Projects/Certifications if "
        "present). NEVER add an empty section or a placeholder like '(no projects "
        "listed)'; if the CV has no projects or certifications, leave that section out.\n"
        "  Write the Summary as 2-3 plain sentences (a short paragraph), NOT bullets.\n"
        "  Use '- ' bullets only under Experience, Projects, and Skills.\n"
        "  Under Experience use '### Role — Company (dates)' followed by '- ' bullets.\n"
        "- For a cover letter: '# Full Name', the contact line, then the letter body as "
        "normal paragraphs.\n\n"
        "Write in Markdown only, never HTML.\n\n"
        f"CONTEXT: {cv_state}"
    )


def _session_tools(session):
    """Tools bound to this session so they can read and write its state."""

    @tool
    def save_document(kind: str, title: str, content: str) -> str:
        """Save a finished CV or cover letter so the user can download it as a file.

        Args:
            kind: 'cv' or 'cover_letter'.
            title: a short title, e.g. "Frontend Developer CV".
            content: the complete document in Markdown.
        """
        normalized = "cover_letter" if "cover" in kind.lower() else "cv"
        content = _clean_doc(content)
        doc_id, docx_path, pdf_path = build_document(normalized, title, content)
        session.documents[doc_id] = Document(
            id=doc_id,
            kind=normalized,
            title=title or normalized,
            content=content,
            docx_path=docx_path,
            pdf_path=pdf_path,
        )
        return f"Document saved and ready to download (id: {doc_id})."

    @tool
    def search_cv(query: str) -> str:
        """Look up relevant details in the user's uploaded CV.

        Args:
            query: what to look for, e.g. "React projects" or "education".
        """
        if session.vector_store is None:
            return "No CV has been uploaded yet."
        return get_relevant_cv_text(session.vector_store, query, k=6)

    return [save_document, search_cv]


def _build_ava(session):
    return create_agent(
        model=get_llm(temperature=0.3),
        tools=get_tools() + _session_tools(session),
        system_prompt=_ava_system_prompt(session),
        checkpointer=_checkpointer,
        name="ava",
    )


def _final_text(result: dict) -> str:
    for msg in reversed(result.get("messages") or []):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "Sorry, I didn't catch that — could you say it again?"


def _error_reply(exc: Exception) -> str:
    message = str(exc)
    if any(k in message for k in ("rate_limit", "429", "413")):
        return "I've hit the free Groq usage limit for a moment. Give it a minute and try again."
    if "recursion" in message.lower():
        return "That took more steps than I expected — could you rephrase or narrow it down?"
    return f"Something went wrong on my end: {message[:200]}"


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_doc_payload(text: str) -> dict | None:
    """Rescue the case where the model printed a save_document payload as text
    instead of calling the tool."""
    for match in _JSON_BLOCK.finditer(text or ""):
        try:
            obj = json.loads(match.group(1))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("content") and (obj.get("kind") or obj.get("title")):
            return obj
    return None


def _strip_json_blocks(text: str) -> str:
    return _JSON_BLOCK.sub("", text or "").strip()


# The gap note belongs in chat, not the document; strip it if it slips into content.
_TRAILING_NOTE = re.compile(r"(?:\n\s*-{3,}\s*)?\n+\s*[>*_ ]*Note\b.*\Z", re.S | re.I)
_H2 = re.compile(r"^\s*##\s+\S")


def _clean_doc(content: str) -> str:
    """Strip a trailing gap note and any empty / '(no ... listed)' section."""
    md = _TRAILING_NOTE.sub("", content or "").rstrip()

    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        if _H2.match(lines[i]):
            j = i + 1
            while j < n and not _H2.match(lines[j]):
                j += 1
            body = "\n".join(lines[i + 1:j]).strip()
            plain = re.sub(r"[*_>#()\-]", "", body).strip()
            empty = (not plain) or re.match(r"(?i)^no\b.*\b(listed|provided|available|specified|found)", plain)
            if empty:
                i = j  # drop this heading and its placeholder body
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip()


def _save_payload(session, payload: dict) -> None:
    kind = "cover_letter" if "cover" in str(payload.get("kind", "")).lower() else "cv"
    title = str(payload.get("title") or kind)
    content = _clean_doc(str(payload.get("content") or ""))
    doc_id, docx_path, pdf_path = build_document(kind, title, content)
    session.documents[doc_id] = Document(
        id=doc_id, kind=kind, title=title, content=content, docx_path=docx_path, pdf_path=pdf_path
    )


def chat_with_ava(session, user_message: str):
    """Run one turn synchronously. Returns (reply_text, [new Document objects])."""
    before = set(session.documents)
    agent = _build_ava(session)
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": session.thread_id}, "recursion_limit": 16},
        )
        reply = _final_text(result)
    except Exception as exc:
        reply = _error_reply(exc)
    # Fallback: model printed the payload instead of calling the tool.
    if not any(i not in before for i in session.documents):
        payload = _extract_doc_payload(reply)
        if payload:
            _save_payload(session, payload)
            reply = _strip_json_blocks(reply)
    new_docs = [session.documents[i] for i in session.documents if i not in before]
    return reply, new_docs


async def stream_ava(session, user_message: str):
    """Stream one turn as ('token', text) chunks, then any ('document', info),
    then ('done', None). Powers the SSE endpoint."""
    before = set(session.documents)
    agent = _build_ava(session)
    buffer = []
    try:
        async for chunk, _meta in agent.astream(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": session.thread_id}, "recursion_limit": 16},
            stream_mode="messages",
        ):
            # Only the model's own tokens; skip tool-result messages.
            if not isinstance(chunk, AIMessageChunk):
                continue
            text = chunk.content
            if isinstance(text, list):
                text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in text
                )
            if text:
                buffer.append(text)
                yield ("token", text)
    except Exception as exc:
        yield ("token", _error_reply(exc))

    # Fallback: model printed the payload instead of calling the tool.
    if not any(i not in before for i in session.documents):
        payload = _extract_doc_payload("".join(buffer))
        if payload:
            _save_payload(session, payload)

    for doc_id in session.documents:
        if doc_id not in before:
            doc = session.documents[doc_id]
            yield ("document", {"id": doc.id, "kind": doc.kind, "title": doc.title})
    yield ("done", None)
