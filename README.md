# Ava — AI Career Coach

Ava is a conversational career coach. Upload your CV and just chat: she tailors it
to a specific job, drafts cover letters, or builds one from scratch — grounding
every edit in your real experience. Finished documents download as **DOCX and PDF**
in a clean, résumé-style layout.

Everything happens through a single chat: replies **stream in live**, you attach a
CV with the paperclip, and Ava **asks before adding anything that isn't already in
your background** — no fabricated skills or dates.

## Stack

- **Backend:** FastAPI (streaming via Server-Sent Events) · LangChain / LangGraph agent · Groq (LLM)
- **RAG:** ChromaDB + FastEmbed (local embeddings) over the uploaded CV
- **Tools:** Tavily (live job / web search)
- **Documents:** python-docx (DOCX) · fpdf2 (PDF)
- **Frontend:** React + Vite · react-markdown

## How it works

Upload a CV → it's chunked and embedded into ChromaDB → you chat with Ava, a
LangGraph agent with per-session memory and tools → she tailors or writes documents
grounded in the CV → `save_document` renders a DOCX and PDF you can download.

```
React chat  ──HTTP / SSE──▶  FastAPI
                              ├─ session / upload   (RAG: ChromaDB + FastEmbed)
                              ├─ message  ─▶ Ava agent (LangGraph + Groq)
                              │               tools: job_search, search_cv, save_document
                              └─ document ─▶ DOCX / PDF
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv ../venv
../venv/Scripts/activate        # Windows  (macOS/Linux: source ../venv/bin/activate)
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys:

- `GROQ_API_KEY` — required ([console.groq.com/keys](https://console.groq.com/keys))
- `TAVILY_API_KEY` — optional, enables job / web search ([app.tavily.com](https://app.tavily.com/home))
- `GROQ_MODEL` — optional; defaults to a sensible model

Run it:

```bash
uvicorn main:app --reload
```

Interactive API docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## API

| Method | Route | Purpose |
| ------ | ----- | ------- |
| POST | `/api/session` | Start a chat; returns a `session_id` and Ava's greeting |
| POST | `/api/upload` | Attach a CV PDF (chunked + embedded for the session) |
| POST | `/api/message` | Send a message; **streams** Ava's reply (SSE) |
| GET | `/api/document/{id}` | Download a generated document (`?fmt=pdf` or `?fmt=docx`) |
| GET | `/api/config` | Which API keys are configured |

## Project layout

```
backend/    FastAPI app, the Ava agent, RAG pipeline, and document generation
frontend/   React (Vite) single-chat UI
uploads/    sample CVs (gitignored)
```

## Notes

- Free-tier Groq / Tavily limits apply; Ava surfaces rate limits gracefully.
- API keys live in `backend/.env` and are gitignored — never commit them.
