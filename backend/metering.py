"""Token usage metering — internal only, never shown to users.

Every model Ava creates comes from llm.get_llm, which attaches one UsageRecorder.
After each successful call the recorder writes one row to a small SQLite ledger:
when, which client and chat session, which feature, which model, tokens in/out
and our estimated cost. No message text is ever stored, and a chat session only
as a one-way hash of its id (see session_ref). A call cut off mid-reply (the
user left) is still billed by Groq, so it is recorded too, with token counts
estimated from the text and estimated=1. Rows are written by one background
thread, so the event loop never waits on SQLite.

Which client and feature a call belongs to comes from context variables set by
the caller (metering.context / metering.feature). anyio's run_in_threadpool and
asyncio/LangGraph tasks copy the current context, so attribution survives the
threadpool and the agent's tool loop. With nothing set, a call is recorded under
client "default" and feature "unknown" — a visible gap in the report, not a loss.

Billing months are calendar months in UTC (the 1st at 05:00 in Pakistan).
"""

import asyncio
import atexit
import contextvars
import functools
import hashlib
import json
import logging
import os
import queue
import re
import secrets
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import anyio
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import get_buffer_string

import config

log = logging.getLogger(__name__)

DEFAULT_CLIENT = "default"
UNKNOWN_FEATURE = "unknown"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_client = contextvars.ContextVar("metering_client", default=DEFAULT_CLIENT)
_session = contextvars.ContextVar("metering_session", default=None)
_feature = contextvars.ContextVar("metering_feature", default=UNKNOWN_FEATURE)


# --- Attribution ----------------------------------------------------------

@contextmanager
def context(client_id=None, session_id=None, feature=None):
    """Attribute model calls inside the block. Only the given fields change."""
    tokens = []
    for var, value in ((_client, client_id), (_session, session_id), (_feature, feature)):
        if value is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except ValueError:
                # Exited from a different context (e.g. a streaming response's
                # generator closed after a client disconnect): nothing to undo there.
                pass


def feature(name):
    return context(feature=name)


def current() -> dict:
    return {"client_id": _client.get(), "session_id": _session.get(), "feature": _feature.get()}


# --- Ledger ---------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    plan        TEXT NOT NULL,
    key_hash    TEXT UNIQUE,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,
    client_id     TEXT NOT NULL,
    session_id    TEXT,
    feature       TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens  INTEGER NOT NULL,
    cost_usd      REAL,
    estimated     INTEGER NOT NULL DEFAULT 0  -- 1: reply cut off, counts estimated
);
CREATE INDEX IF NOT EXISTS usage_client_ts ON usage (client_id, ts_utc);
"""

_init_lock = threading.Lock()
_initialized: set[str] = set()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _now() -> str:
    return _iso(datetime.now(timezone.utc))


def _db_path() -> str:
    path = config.METERING_DB
    return path if os.path.isabs(path) else os.path.join(_BASE_DIR, path)


def _has_estimated(conn: sqlite3.Connection) -> bool:
    return "estimated" in {col[1] for col in conn.execute("PRAGMA table_info(usage)")}


def _connect() -> sqlite3.Connection:
    """A fresh connection per operation (safe across threads), schema ensured once per path."""
    path = _db_path()
    with _init_lock:
        if path not in _initialized:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with closing(sqlite3.connect(path, timeout=5)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                # Ledgers from before cut-off replies were recorded lack this column.
                if not _has_estimated(conn):
                    try:
                        conn.execute("ALTER TABLE usage ADD COLUMN estimated INTEGER NOT NULL DEFAULT 0")
                    except sqlite3.OperationalError:
                        # Another process (the CLI, a second worker) added it first.
                        if not _has_estimated(conn):
                            raise
                conn.execute(
                    "INSERT OR IGNORE INTO clients (client_id, name, plan, key_hash, created_utc) "
                    "VALUES (?, ?, ?, NULL, ?)",
                    (DEFAULT_CLIENT, "Default (no client key)", "internal", _now()),
                )
                conn.commit()
            _initialized.add(path)
    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


# --- Plans and prices -----------------------------------------------------

_plans: dict | None = None
_prices: dict | None = None
_warned_models: set[str] = set()


def _load_json(name: str) -> dict:
    try:
        with open(os.path.join(_BASE_DIR, name), encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    except Exception:
        log.error("metering: could not load %s", name, exc_info=True)
        return {}


def plans() -> dict:
    global _plans
    if _plans is None:
        _plans = _load_json(config.PLANS_FILE)
    return _plans


def prices() -> dict:
    global _prices
    if _prices is None:
        _prices = _load_json(config.PRICING_FILE)
    return _prices


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Our estimated USD cost, or None (never 0) when the model has no usable price entry."""
    price = prices().get(model)
    try:
        return round((input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000, 8)
    except (KeyError, TypeError):
        # Missing or malformed entry: the tokens are still recorded, just unpriced.
        if model not in _warned_models:
            _warned_models.add(model)
            log.warning("metering: no price for model %r in %s; cost recorded as null",
                        model, config.PRICING_FILE)
        return None


# --- Recording ------------------------------------------------------------

def session_ref(session_id: str | None) -> str | None:
    """What the ledger stores for a chat session: a one-way hash, never the id.

    A session id alone opens a chat (and the CV uploaded to it), so it must not
    sit in billing data. The hash still groups a chat's calls. Session ids are
    random UUIDs, so, as with client keys, a plain SHA-256 can't be reversed.
    """
    return None if session_id is None else hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def record(provider: str, model: str, input_tokens: int, output_tokens: int, estimated: bool = False,
           *, ctx: dict | None = None, ts: str | None = None) -> None:
    """Write one usage row now. Raises on DB errors.

    Attributed from the current context and timed now, unless the caller
    captured ctx (see current()) and ts earlier, as the recorder does.
    """
    ctx = ctx or current()
    inp, out = int(input_tokens), int(output_tokens)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO usage (ts_utc, client_id, session_id, feature, provider, model, input_tokens, "
            "output_tokens, total_tokens, cost_usd, estimated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ts or _now(), ctx["client_id"], session_ref(ctx["session_id"]), ctx["feature"], provider, model,
             inp, out, inp + out, estimate_cost(model, inp, out), int(estimated)),
        )


# Rows from model calls go through one writer thread: callbacks can run on the
# event loop, which must never wait on SQLite (a locked ledger would freeze
# every chat for the busy timeout), and one writer means our own rows never
# contend for the ledger's write lock.
_pending: queue.Queue = queue.Queue()
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()


def _write_pending() -> None:
    while True:
        args, kwargs = _pending.get()
        try:
            record(*args, **kwargs)
        except Exception:
            log.warning("metering: could not record usage", exc_info=True)
        finally:
            _pending.task_done()


def _enqueue(provider: str, model: str, input_tokens: int, output_tokens: int, estimated: bool = False,
             ctx: dict | None = None) -> None:
    """Queue one row for the writer, attributed (ctx, default: the current context)
    and timed now. Returns at once."""
    global _writer
    with _writer_lock:
        if _writer is None:
            _writer = threading.Thread(target=_write_pending, name="metering-writer", daemon=True)
            _writer.start()
    _pending.put(((provider, model, input_tokens, output_tokens, estimated),
                  {"ctx": ctx or current(), "ts": _now()}))


def flush() -> None:
    """Block until every queued row is written (or has failed and been logged).
    For tests and shutdown; never call it on the event loop."""
    _pending.join()


atexit.register(flush)


def _extract_usage(response) -> tuple[int, int] | None:
    """(input, output) tokens from a LangChain LLMResult, or None if it carries none.

    Prefers each generation's message.usage_metadata (set for both plain and
    streamed ChatGroq replies); falls back to llm_output['token_usage'].
    """
    inp = out = 0
    found = False
    for generations in response.generations:
        for gen in generations:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                inp += int(meta.get("input_tokens", 0))
                out += int(meta.get("output_tokens", 0))
                found = True
    if not found:
        usage = (response.llm_output or {}).get("token_usage") or {}
        if usage:
            inp, out = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
            found = True
    return (inp, out) if found else None


_CHARS_PER_TOKEN = 4   # rough, for calls cut off before the provider's count arrived


def _estimate_tokens(chars: int) -> int:
    return -(-chars // _CHARS_PER_TOKEN)   # rounded up


def _output_chars(token, message) -> int:
    """Characters of output in one streamed chunk: text, reasoning and tool-call arguments."""
    chars = len(token) if isinstance(token, str) else 0
    if message is not None:
        chars += len(getattr(message, "additional_kwargs", {}).get("reasoning_content") or "")
        chars += sum(len(c.get("args") or "") for c in getattr(message, "tool_call_chunks", None) or [])
    return chars


def _is_cancellation(error: BaseException) -> bool:
    try:
        anyio_cancelled = anyio.get_cancelled_exc_class()
    except Exception:   # no async library running in this thread
        anyio_cancelled = asyncio.CancelledError
    return isinstance(error, (asyncio.CancelledError, GeneratorExit, anyio_cancelled))


@dataclass
class _Call:
    """A model call in flight, as far as the recorder knows it."""
    ctx: dict                               # attribution, read when the call started
    input_tokens: int                       # estimated from the prompt
    output_chars: int = 0                   # streamed so far
    answered: bool = False                  # a chunk arrived: Groq accepted the request
    usage: tuple[int, int] | None = None    # exact, once a chunk carries it
    task: asyncio.Task | None = None        # the async task running the call
    on_task_done: Callable | None = None


class UsageRecorder(BaseCallbackHandler):
    """Writes one ledger row per model call that Groq bills.

    Attached once, through the model constructor in llm.get_llm — never also at
    invoke time, so each call is counted exactly once. A finished call is
    recorded with the provider's exact counts. A call cut off mid-reply is
    recorded as an estimate (see _cut_off). A failed request (HTTP error, rate
    limit) isn't billed, so it isn't recorded; the client retries those itself.

    run_inline: every callback runs synchronously where LangChain raises it, in
    the caller's context. Nothing here awaits, so a cancellation can't interrupt
    recording, and rows are only queued (see _enqueue), so the event loop never
    waits on SQLite.
    """

    raise_error = False   # a metering failure must never break a chat
    run_inline = True

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._calls: dict = {}   # run_id -> _Call

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        chars = sum(len(get_buffer_string(batch)) for batch in messages)
        tools = (kwargs.get("invocation_params") or {}).get("tools")
        if tools:
            chars += len(json.dumps(tools))
        call = self._calls[run_id] = _Call(current(), _estimate_tokens(chars))
        try:
            call.task = asyncio.current_task()
        except RuntimeError:   # a sync call on a worker thread: it always ends or errors
            return
        if call.task is not None:
            # When the task running an async call is cancelled mid-call (the
            # agent's step, once a reply is abandoned), LangChain may report
            # neither the end nor an error. A call still open when its task
            # finishes was cut off.
            call.on_task_done = functools.partial(self._task_done, run_id)
            call.task.add_done_callback(call.on_task_done)

    def on_llm_new_token(self, token, *, chunk=None, run_id=None, **kwargs) -> None:
        call = self._calls.get(run_id)
        if call is None:
            return
        message = getattr(chunk, "message", None)
        call.answered = True
        call.output_chars += _output_chars(token, message)
        usage = getattr(message, "usage_metadata", None)
        if usage:
            call.usage = (int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        call = self._end(run_id)
        try:
            usage = _extract_usage(response)
            if usage is None:
                log.warning("metering: %s reply carried no token usage; not recorded", self.model)
                return
            _enqueue(self.provider, self.model, *usage, ctx=call.ctx if call else None)
        except Exception:
            log.warning("metering: could not record usage", exc_info=True)

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        call = self._end(run_id)
        if call is not None and _is_cancellation(error):
            self._cut_off(call)

    def _end(self, run_id) -> "_Call | None":
        """Forget a call that has ended or failed; returns what was known about it."""
        call = self._calls.pop(run_id, None)
        if call is not None and call.task is not None:
            call.task.remove_done_callback(call.on_task_done)
        return call

    def _task_done(self, run_id, task) -> None:
        call = self._calls.pop(run_id, None)
        if call is not None:
            self._cut_off(call)

    def _cut_off(self, call: _Call) -> None:
        """Record a call cut off mid-reply: Groq bills what it generated. Exact if
        the provider's count arrived before the cut, else estimated from the text.

        A call cut off before its first chunk isn't recorded: Groq may never have
        accepted it (the client was still connecting, or waiting to retry a 429)."""
        if not call.answered:
            return
        if call.usage is not None:
            _enqueue(self.provider, self.model, *call.usage, ctx=call.ctx)
        else:
            _enqueue(self.provider, self.model, call.input_tokens, _estimate_tokens(call.output_chars),
                     estimated=True, ctx=call.ctx)


# --- Clients --------------------------------------------------------------

def _hash_key(key: str) -> str:
    # SHA-256, deliberately not bcrypt: client keys are 256-bit random strings we
    # generate (not human passwords), so there is nothing to brute-force and a
    # slow hash would only add latency to every session start.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _require_plan(plan: str) -> None:
    if plan not in plans():
        raise ValueError(f"unknown plan {plan!r}; choose from: {', '.join(plans())}")


def add_client(name: str, plan: str) -> tuple[str, str]:
    """Create a client. Returns (client_id, key); the key is shown once and stored only hashed."""
    _require_plan(plan)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "client"
    client_id = f"{slug}-{secrets.token_hex(3)}"
    key = "ava_" + secrets.token_urlsafe(32)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO clients (client_id, name, plan, key_hash, created_utc) VALUES (?,?,?,?,?)",
            (client_id, name, plan, _hash_key(key), _now()),
        )
    return client_id, key


def client_for_key(key: str | None) -> str | None:
    if not key:
        return None
    with closing(_connect()) as conn:
        row = conn.execute("SELECT client_id FROM clients WHERE key_hash = ?", (_hash_key(key),)).fetchone()
    return row["client_id"] if row else None


def set_plan(client_id: str, plan: str) -> None:
    _require_plan(plan)
    with closing(_connect()) as conn, conn:
        if conn.execute("UPDATE clients SET plan = ? WHERE client_id = ?", (plan, client_id)).rowcount == 0:
            raise ValueError(f"unknown client {client_id!r}")


def list_clients() -> list[dict]:
    with closing(_connect()) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT client_id, name, plan, created_utc FROM clients ORDER BY created_utc")]


# --- Allowances -----------------------------------------------------------

def month_bounds(month: str | None = None) -> tuple[str, str, str]:
    """('YYYY-MM', start, end) of a UTC calendar month; defaults to the current one."""
    if month is None:
        now = datetime.now(timezone.utc)
        year, mon = now.year, now.month
    else:
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month or ""):
            raise ValueError("month must look like YYYY-MM")
        year, mon = int(month[:4]), int(month[5:])
    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    end = datetime(year + (mon == 12), mon % 12 + 1, 1, tzinfo=timezone.utc)
    return f"{year:04d}-{mon:02d}", _iso(start), _iso(end)


def _plan_limit(plan: str) -> int | None:
    if plan not in plans():
        log.warning("metering: unknown plan %r; treating it as uncapped", plan)
        return None
    return plans()[plan].get("monthly_tokens")


@dataclass(frozen=True)
class Allowance:
    allowed: bool
    used: int
    limit: int | None   # None = uncapped


def check_allowance(client_id: str) -> Allowance:
    """Tokens used this UTC month vs the client's plan. Fails open on any error."""
    try:
        _, start, end = month_bounds()
        with closing(_connect()) as conn:
            row = conn.execute("SELECT plan FROM clients WHERE client_id = ?", (client_id,)).fetchone()
            used = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM usage "
                "WHERE client_id = ? AND ts_utc >= ? AND ts_utc < ?",
                (client_id, start, end),
            ).fetchone()[0]
        if row is None:
            log.warning("metering: unknown client %r; treating it as uncapped", client_id)
            return Allowance(True, used, None)
        limit = _plan_limit(row["plan"])
        return Allowance(limit is None or used < limit, used, limit)
    except Exception:
        log.error("metering: allowance check failed; allowing the turn", exc_info=True)
        return Allowance(True, 0, None)


# --- Report ---------------------------------------------------------------

_REPORT_NOTE = (
    "Token counts are exact, as reported by the model provider, except for replies cut off "
    "mid-stream (e.g. the user left), which are recorded as estimates (estimated=1, about 4 "
    "characters per token; see estimated_calls). Costs are estimates from pricing.json "
    "(prompt-cache discounts ignored); null means a model has no price entry (a client's "
    "total is null if any of its calls is unpriced; see unpriced_calls)."
)


def usage_report(month: str | None = None, client_id: str | None = None) -> dict:
    """Usage for one UTC month: per-client totals plus a client/feature/model/day breakdown."""
    label, start, end = month_bounds(month)
    where, args = "ts_utc >= ? AND ts_utc < ?", [start, end]
    if client_id:
        where += " AND client_id = ?"
        args.append(client_id)
    with closing(_connect()) as conn:
        breakdown = [dict(r) for r in conn.execute(
            "SELECT client_id, feature, model, substr(ts_utc, 1, 10) AS day, COUNT(*) AS calls, "
            "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, "
            "SUM(cost_usd IS NULL) AS unpriced_calls, SUM(estimated) AS estimated_calls "
            f"FROM usage WHERE {where} GROUP BY client_id, feature, model, day "
            "ORDER BY day, client_id, feature, model", args)]
        totals = {r["client_id"]: dict(r) for r in conn.execute(
            "SELECT client_id, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, "
            "SUM(cost_usd IS NULL) AS unpriced_calls, SUM(estimated) AS estimated_calls "
            f"FROM usage WHERE {where} GROUP BY client_id", args)}
        known = [dict(r) for r in conn.execute(
            "SELECT client_id, name, plan FROM clients" + (" WHERE client_id = ?" if client_id else "")
            + " ORDER BY created_utc", [client_id] if client_id else [])]
    # Usage under a client id with no account row still has to show up in the
    # totals, or it would be missing from the bill.
    registered = {row["client_id"] for row in known}
    known += [{"client_id": cid, "name": None, "plan": None} for cid in totals if cid not in registered]
    clients = []
    for row in known:
        t = totals.get(row["client_id"], {})
        used = t.get("total_tokens") or 0
        unpriced = t.get("unpriced_calls") or 0
        limit = None if row["plan"] is None else _plan_limit(row["plan"])
        clients.append({
            "client_id": row["client_id"], "name": row["name"], "plan": row["plan"],
            "monthly_tokens": limit,
            "used_tokens": used,
            "remaining_tokens": None if limit is None else max(limit - used, 0),
            "input_tokens": t.get("input_tokens") or 0,
            "output_tokens": t.get("output_tokens") or 0,
            # null when any unpriced call leaves the cost unknown (SUM would count it
            # as $0); no calls at all is $0.
            "cost_usd": None if unpriced else (t.get("cost_usd") or 0.0),
            "unpriced_calls": unpriced,
            "estimated_calls": t.get("estimated_calls") or 0,
        })
    return {
        "month": label, "timezone": "UTC",
        "period_start_utc": start, "period_end_utc": end,
        "note": _REPORT_NOTE,
        "clients": clients,
        "breakdown": breakdown,
    }
