# Ava — AI Career Coach

Ava is a conversational career coach. Upload your CV and just chat: she tailors it
to a specific job, drafts cover letters, or builds one from scratch — grounding
every edit in your real experience. Finished documents download as **DOCX and PDF**
in a clean, résumé-style layout.

Everything happens through a single chat: replies **stream in live**, you attach a
CV with the paperclip, and Ava **asks before adding anything that isn't already in
your background** — no fabricated skills or dates.

Ava can also run a **mock interview** for a specific job: paste a job description
(or share a link) and she asks role-specific questions one at a time, scores each
answer with short feedback, and finishes with a readiness report. Type `skip`,
`repeat` or `end interview` at any time.

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
                              │               tools: job_search, web_search, search_cv, save_document,
                              │                      propose_email, start_mock_interview, MCP fetch
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

### MCP tools (optional)

Ava can use external [MCP](https://modelcontextprotocol.io) servers. Install
[`uv`](https://docs.astral.sh/uv/) (provides `uvx`), then copy
`backend/mcp_servers.example.json` to `backend/mcp_servers.json`. The example
enables the **Fetch** server, so Ava can read a job posting from a URL and tailor
against it.

Notes: `uvx` must be on the PATH of the process running the backend. Fetch
respects `robots.txt` and can't run JavaScript, so static pages (Greenhouse,
Lever, Workable, most company career pages) work well; LinkedIn/Indeed/Glassdoor
often don't — paste the text in those cases. Each fetch call spawns the server
(~1–3s). If MCP isn't configured, Ava runs exactly as before.

### Usage metering (internal)

Ava records every LLM call in a local SQLite ledger, `backend/data/usage.db`
(gitignored), so clients can be billed by tokens or sold monthly allowances. Each
row holds the time, client, chat session, feature (`chat`, `guard`, `interview.*`),
model, input/output tokens and an estimated cost — **never message text**. The chat
session is stored as a one-way hash, because a session id on its own opens that
chat. Users never see usage.

**Clients and keys.** From `backend/`, create a client; its key is printed once
and stored only as a hash:

```bash
python manage_clients.py add "Acme Ltd" --plan starter
python manage_clients.py list
python manage_clients.py set-plan <client_id> pro
python manage_clients.py plans
```

Put the key in `frontend/.env` as `VITE_AVA_CLIENT_KEY=ava_...` and the frontend
sends it as an `X-Client-Key` header when a chat starts. The key ships in the
browser bundle, so it identifies the client — it doesn't authenticate users. Chats
without a key count under the built-in, uncapped `default` client; set
`REQUIRE_CLIENT_KEY=true` in `backend/.env` to reject them instead.

**Plans.** Monthly token allowances (input + output) live in `backend/plans.json`:
`internal` and `payg` are uncapped, `free` / `starter` / `pro` have placeholder
limits, and `suspended` blocks a client without deleting its history. At the limit,
Ava tells the user their organisation has used this month's allowance and makes no
model call; a turn that has already started may finish, so a client can go slightly
over. Each turn in flight holds a share of what's left (`TURN_TOKEN_RESERVE`, 20,000
tokens by default), so a burst of parallel chats can't all start on the last few
tokens; near the limit, the extra chats are asked to try again in a moment.
Months are calendar months in **UTC** (they roll over at 05:00 on the 1st in
Pakistan).

**Cost.** `backend/pricing.json` holds our USD price per 1M tokens for each model.
Token counts are exact (reported by Groq); cost is an estimate, and a model missing
from the file is recorded with cost `null` (unpriced), never 0. Restart the backend
after editing either JSON file.

**Report.** Set `ADMIN_API_KEY` in `backend/.env` to enable
`GET /api/admin/usage?month=YYYY-MM` (optionally `&client_id=...`) with an
`X-Admin-Key` header. It returns per-client totals and a breakdown by feature,
model and day, and stays closed (404) while the key is unset. The same totals from
the command line:

```bash
python manage_clients.py usage                       # current month
python manage_clients.py usage --month 2026-09 --client <client_id>
```

## API

| Method | Route | Purpose |
| ------ | ----- | ------- |
| POST | `/api/session` | Start a chat; returns a `session_id` and Ava's greeting (optional `X-Client-Key` header) |
| POST | `/api/upload` | Attach a CV PDF (chunked + embedded for the session) |
| POST | `/api/message` | Send a message; **streams** Ava's reply (SSE) |
| GET | `/api/document/{id}` | Download a generated document (`?fmt=pdf` or `?fmt=docx`) |
| GET | `/api/config` | Which API keys are configured |
| GET | `/api/admin/usage` | Internal token-usage report for a UTC month (`X-Admin-Key`; `?month=YYYY-MM&client_id=`) |

## Testing

Backend tests run with **pytest**. From `backend/` with the virtualenv active:

```bash
pytest
```

Most tests are offline and fast. The prompt-injection suite also has a few
end-to-end checks that call the live LLM; they're skipped by default and run
only when you opt in:

```bash
RUN_LLM_TESTS=1 pytest test_injection.py
```

On Windows PowerShell, set the flag first: `$env:RUN_LLM_TESTS=1; pytest test_injection.py`.

## Project layout

```
backend/    FastAPI app, the Ava agent, RAG pipeline, and document generation
frontend/   React (Vite) single-chat UI
uploads/    sample CVs for local testing (gitignored; runtime uploads are parsed in memory, not stored)
```