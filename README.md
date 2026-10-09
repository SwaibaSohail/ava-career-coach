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
row holds the time, client, chat session, feature (`chat`, `guard`, `guard.shadow`, `interview.*`, `eval.guard`),
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
`REQUIRE_CLIENT_KEY=true` in `backend/.env` to reject them instead. With keys
required, a chat can't start while the ledger can't be read (the session request
gets a 503 and can be retried); without, it starts under `default` and the error
is logged.

**Plans.** Monthly token allowances (input + output) live in `backend/plans.json`:
`internal` and `payg` are uncapped, `free` / `starter` / `pro` have placeholder
limits, and `suspended` blocks a client without deleting its history. The allowance
is checked once per message, before any model call. At the limit, Ava tells the
user their organisation has used this month's allowance and makes no model call. A
message that has passed the check is allowed to finish, so a client can go slightly
over: one long reply can cross the limit, and several chats sent at the same moment
near the limit can each pass the check and each finish.
Months are calendar months in **UTC** (they roll over at 05:00 on the 1st in
Pakistan).

**Cost.** `backend/pricing.json` holds our USD price per 1M tokens for each model.
Token counts are exact (reported by Groq); cost is an estimate, and a model missing
from the file is recorded with cost `null` (unpriced), never 0; a client's monthly
cost is `null` if any of its calls is unpriced. Restart the backend
after editing either JSON file.

**Cut-off replies.** If a model call is cut off mid-reply (usually because the user
closed the chat), Groq still bills what it generated, but its token count, which
comes at the end of the reply, never arrives. Such a call is recorded with
`estimated = 1`: input tokens from the prompt's length, output tokens from the text
streamed so far, at about 4 characters per token. A request that fails outright (an
HTTP error or rate limit) isn't billed by Groq and isn't recorded, nor is one cut
off before Groq starts answering (e.g. while waiting to retry a rate limit). Older ledgers
gain the `estimated` column automatically.

**Report.** Set `ADMIN_API_KEY` in `backend/.env` to enable
`GET /api/admin/usage?month=YYYY-MM` (optionally `&client_id=...`) with an
`X-Admin-Key` header. It returns per-client totals and a breakdown by feature,
model and day, each with `estimated_calls` (cut-off replies). It stays closed while
the key is unset: every method gets a 404, as for a page that doesn't exist. The
same totals from the command line, which also flags estimated and unpriced calls:

```bash
python manage_clients.py usage                       # current month
python manage_clients.py usage --month 2026-09 --client <client_id>
```

### Clef guard (optional)

Every chat message goes through cheap checks first (length, a prompt-injection
regex, abuse and spam rules). Only messages that pass reach stage 5, a model that
classifies the message as clean, injection, abuse or harmful. By default that is
a small Groq model (`GUARD_MODEL`) answering with one label. Stage 5 can instead
run on Cloudflare's **Clef**, which returns a probability for each class rather
than a label. Why:

- **A threshold we can tune.** With a probability per class, how many real
  messages get wrongly blocked is set from data (`CLEF_BLOCK_THRESHOLD`). A label
  can't be tuned.
- **A separate rate limit.** The guard stops sharing Groq's limit with the chat.
- **Reuse.** `backend/clef.py` is a general decision client that Alfred can use
  later (routing, lead scoring, handoff).

**Modes** (`GUARD_BACKEND`):

- `groq` (default): the Groq guard decides, as before.
- `shadow`: Groq decides and its answer is used at once. Clef is asked the same
  question in the background (two workers; skipped while 20 are waiting) and its
  answer is only logged, so the two can be compared on real traffic. A reply is
  never held up waiting for Clef.
- `clef`: Clef decides.

`shadow` and `clef` need `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (a
Workers AI token: dashboard → AI → Workers AI → Use REST API). Without them the
backend logs a warning once and runs as `groq`. The token is never logged and
`/api/config` doesn't expose it. A token pasted with a stray space, quote or
invisible character is refused before any request, and logged at ERROR like a
rejected one.

**How Clef decides.** One request asks four questions, versioned together as
`guard-v1`: which of the four classes the message is (a choice), and a yes/no
question for each bad class. With `CLEF_GUARD_RULE=choice` (the default), a
message is blocked when the most likely bad class reaches `CLEF_BLOCK_THRESHOLD`
(default 0.4), and that class is the label. With `noul`, the yes/no answers decide
the same way. Ties go injection, then abuse, then harmful. Under `choice`, a
message whose risk is split across classes (0.3 each) is not blocked, because no
single class reaches the threshold. The rule and threshold were picked by the
offline side-by-side against Groq on question set `guard-v1`; a new
question-set version needs a new sweep before the threshold is trusted.

**Long messages.** By default (`CLEF_WINDOW_CHARS=0`) every message is sent to
Clef whole: the cutoff probe caught a trigger at every position in a
10,000-character CV, longer than Ava's 8,000-character cap. Should Workers AI
start truncating the state, set `CLEF_WINDOW_CHARS` (for example 6000) and a
message longer than that is checked as two windows, its first and last that many
characters, sent in parallel; each bad class then takes the higher of its two
probabilities, so a trigger in either window counts.

**When Clef fails** (a timeout, a server error, a rejected token, the daily
allowance used up): with `CLEF_FALLBACK=groq` (the default), the Groq guard runs on
a tight budget instead: `CLEF_FALLBACK_TIMEOUT_S` (5 s) and no retries. If that
fails too, the check is skipped and the message goes through. Any other error in
the Clef check falls back the same way. The chain is Clef (3 s for the whole
check, `CLEF_TIMEOUT_S`) → Groq (5 s) → skip, so the guard waits about 8 s at
most. In Groq-only mode it waits up to 8 s per attempt over 2 attempts, half a
second apart; it never waits out a rate limit's `Retry-After`. Each limit is a
deadline the guard keeps itself, however slowly an answer trickles in (httpx's
own timeout applies to each read, not to the whole call). With
`CLEF_FALLBACK=open`, a failed Clef check is skipped straight away (the regex and
rule stages still apply).

**Daily allowance.** Clef runs on Workers AI's free 10,000 Neurons a day, shared
by everything on the Cloudflare account. Once they are used up, Clef isn't called
again until they reset at **00:00 UTC (05:00 PKT)**. This is logged once at ERROR,
and the fallback answers in the meantime. In the first 10 minutes after 00:00 UTC,
a used-up answer means Cloudflare hasn't reset yet (the clocks differ by seconds,
or the call was sent before midnight), so Clef is tried again a minute later
rather than paused for another day. A rejected token is logged at ERROR at
most every 5 minutes. Clef calls go into the usage ledger as provider `cloudflare`,
feature `guard` (`guard.shadow` in shadow mode), one row per window.

**Decision log.** Every Clef guard check writes one row to `backend/data/guard.db`
(`GUARD_LOG_DB`, gitignored): the time, mode, question-set version, rule,
threshold, model, number of windows, Clef's seven probabilities, Clef's label,
Groq's label (in shadow mode; empty when the Groq check failed), latency and the
error kind, if any. It never holds the message. Rows are written by a background
thread, so a slow or locked log never holds up a reply. The rule and threshold
can be re-tuned from these rows.

**Shadow text (dev only).** With `GUARD_SHADOW_STORE_TEXT=true`, shadow mode also
keeps the text of messages where Clef and Groq disagreed, so they can be read and
labelled. A message Groq failed to check is not a disagreement. These rows are
deleted after `GUARD_SHADOW_RETENTION_DAYS` (14), whatever mode the guard runs in
by then: the backend purges them at startup, every hour and before listing them,
and overwrites them on disk. The flag is off by default and should stay off in
production, where shadow mode then gives agreement rates, not examples.

**Privacy.** Messages already go to Groq for the guard. Clef adds Cloudflare as a
second processor of the same text; Cloudflare states that it does not store
Workers AI inputs or train on them. Disagreement text, when enabled, stays on our
server.

| Setting (`backend/.env`) | Default | |
| ------------------------ | ------- | - |
| `GUARD_BACKEND` | `groq` | `groq`, `shadow` or `clef` |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | empty | needed for `shadow` and `clef` |
| `CLEF_GUARD_MODEL` | `clef` | or `clef-flash` (cheaper) |
| `CLEF_GUARD_RULE` | `choice` | or `noul` |
| `CLEF_BLOCK_THRESHOLD` | `0.4` | the probability that blocks (tuned for `guard-v1`, `choice`) |
| `CLEF_WINDOW_CHARS` | `0` | one window; e.g. `6000` = first and last 6,000 characters |
| `CLEF_TIMEOUT_S` | `3` | seconds for the whole Clef check, no retries |
| `CLEF_FALLBACK` | `groq` | or `open` |
| `CLEF_FALLBACK_TIMEOUT_S` | `5` | the Groq fallback's budget, no retries |
| `GUARD_LOG_DB` | `data/guard.db` | resolved against `backend/` |
| `GUARD_SHADOW_STORE_TEXT` | `false` | dev only |
| `GUARD_SHADOW_RETENTION_DAYS` | `14` | |

Restart the backend after changing them.

### Guard side-by-side eval (offline)

`backend/evals/guard_eval.py` compares the Groq guard with Clef and Clef-flash on
labelled messages, through the same code the chat uses. It is run by hand with the
real Groq and Cloudflare keys and is not part of `pytest`. From `backend/`:

```bash
python -m evals.guard_eval cutoff                    # how much of a long message Clef reads
python -m evals.guard_eval cutoff --lang ur          # the same with an Urdu-script CV
python -m evals.guard_eval run --set dev --backends groq,clef,clef-flash
python -m evals.guard_eval score --set dev --sweep   # pick the rule and threshold
python -m evals.guard_eval score --set holdout --rule choice --threshold 0.4
python -m evals.guard_eval disagreements --export unlabelled.jsonl
```

**Sets.** `--set dev` and `--set holdout` read `backend/evals/guard_dev.jsonl` and
`backend/evals/guard_holdout.jsonl`; any other path works too. Each line is one
message: `{"id", "text", "label", "lang", "kind"}`, with `label` one of `clean`,
`injection`, `abuse`, `harmful` and `lang` one of `en`, `ur`, `roman_ur`, `mixed`.
A bad row stops the command and names its line. Tune on dev; report on holdout,
which stays frozen. No two rows of one label are versions of the same text, and
the holdout shares no text with dev (both are checked by `pytest`).

**cutoff** sends a clean 10,000-character CV with a blunt injection placed after
500, 1,000, 2,000 and so on up to 9,500 characters, each text whole as one window
(12 calls). It prints Clef's `p(injection)`, `is_injection` and input tokens at each
offset, and the offset up to which every one was caught; a miss before a later
catch is flagged, and only the unbroken run counts. Run it with `--lang ur` too:
Urdu script takes more tokens a character, so if Clef reads a set number of
tokens it reads fewer Urdu characters. Only if both runs catch the trigger at
every offset past 8,000 characters (Ava's cap) does one window read every message
(`CLEF_WINDOW_CHARS=0`). Otherwise set `CLEF_WINDOW_CHARS` no higher than the
smaller of the two offsets; the first and last windows cover a whole
8,000-character message only while the window is at least 4,000 characters.

**run** asks each backend about each row, one call at a time, and caches the
answers in `backend/evals/out/` (gitignored), one file per set, backend and
question-set version. A later run asks only what isn't cached yet: a new row, an
edited row, every Groq row after `GUARD_MODEL`, `GUARD_REASONING_EFFORT` or the
guard prompt changes, and the Clef rows a new `CLEF_WINDOW_CHARS` splits
differently (or every Clef row after the model changes); `--fresh` asks again
anyway and `--limit N` takes the first N rows. Clef's seven probabilities are kept
for each window, so any rule and threshold can be scored later without new calls.
Stage 5 is asked about every message, even one stages 1-4 would block. Before
calling Clef, `run` prints the estimated Neurons against the free 10,000 a day and
won't use more than 80% of them without `--yes`. If the allowance runs out it stops, and the answers so far
stay cached. Don't run the eval on the same UTC day as a day of shadow traffic.
Calls are billed in the usage ledger as feature `eval.guard`.

**score** makes no calls. For each backend it prints the verdicts against the true
labels; bad messages caught and missed, and good ones wrongly blocked, overall, by
language and by kind; latency p50, p95 and max; errors by kind; mean input
tokens; and USD per 1,000 messages. It also gives the counts with stages 1-4 in
front, worked out as it scores, so a regex or rule edit counts at once. A Clef
check that failed counts as letting the message through. Only answers that fit
the settings as they are now are scored; rows without one are counted and left
out. `--rule` and `--threshold` (default: the configured ones) re-score Clef from
the cache; `--sweep` prints caught/missed/wrongly blocked at thresholds 0.30 to
0.95 under both rules.

**Reading the acceptance block.** Each Clef backend is compared with Groq on the
same rows. Each check prints `ok` or `FAIL` with both counts.

- `PASS`: Clef catches at least Groq's catches minus 2, wrongly blocks at most
  Groq's plus 2 overall and Groq's plus 1 in every language, has a p95 latency
  under `CLEF_TIMEOUT_S`, and times out on under 1% of messages.
- `clearly not worse: YES`: Clef catches at least as many as Groq and wrongly
  blocks no more.
- Switch `GUARD_BACKEND` to `clef` only when both hold on the holdout set; the last
  line then gives the settings to use. Otherwise keep `groq`, share the numbers,
  and run shadow mode to collect real disagreements. While any row is left out
  (a run cut short by `--limit` or the daily allowance), the block is marked
  provisional and never recommends the switch: the holdout is sorted by label, so
  the missing rows may be whole classes. These are counts on about 130 messages: a
  smoke test, not a benchmark.

**Result so far (2026-10-09, `guard-v1`, `choice`, 0.4).** On the frozen holdout
Clef passed but was not clearly not worse (74/75 caught against Groq's 71, one
wrong block against Groq's none, p50 0.47 s against 2.7 s), so `groq` stays the
default and the next step is shadow mode on real traffic.

**disagreements** lists what shadow mode stored (only with
`GUARD_SHADOW_STORE_TEXT=true`). `--export` writes them as set rows with `label`
and `lang` left empty, to be labelled before they join a set. A relative path goes
in `backend/evals/out/` (gitignored), since the file holds real messages that the
retention purge can't reach: delete it once they are labelled.

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