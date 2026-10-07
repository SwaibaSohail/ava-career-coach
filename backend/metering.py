"""Token usage metering — internal only, never shown to users.

Every model Ava creates comes from llm.get_llm, which attaches one UsageRecorder.
After each successful call the recorder writes one row to a small SQLite ledger:
when, which client and chat session, which feature, which model, tokens in/out
and our estimated cost. No message text is ever stored, and a chat session only
as a one-way hash of its id (see session_ref).

Which client and feature a call belongs to comes from context variables set by
the caller (metering.context / metering.feature). anyio's run_in_threadpool and
asyncio/LangGraph tasks copy the current context, so attribution survives the
threadpool and the agent's tool loop. With nothing set, a call is recorded under
client "default" and feature "unknown" — a visible gap in the report, not a loss.

Billing months are calendar months in UTC (the 1st at 05:00 in Pakistan).
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

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
    cost_usd      REAL
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


def _connect() -> sqlite3.Connection:
    """A fresh connection per operation (safe across threads), schema ensured once per path."""
    path = _db_path()
    with _init_lock:
        if path not in _initialized:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with closing(sqlite3.connect(path, timeout=5)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
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


def record(provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Write one usage row attributed from the current context. Raises on DB errors."""
    ctx = current()
    inp, out = int(input_tokens), int(output_tokens)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO usage (ts_utc, client_id, session_id, feature, provider, model, "
            "input_tokens, output_tokens, total_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_now(), ctx["client_id"], session_ref(ctx["session_id"]), ctx["feature"], provider, model,
             inp, out, inp + out, estimate_cost(model, inp, out)),
        )


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


class UsageRecorder(BaseCallbackHandler):
    """Writes one ledger row per successful model call.

    Attached once, through the model constructor in llm.get_llm — never also at
    invoke time, so each call is counted exactly once. Failed calls (which the
    client retries) never reach on_llm_end.

    Not run_inline: on async calls (the agent's turns) LangChain then runs this
    in its executor under a copy of the caller's context, so attribution is kept
    and a locked ledger can't stall the event loop for the busy timeout. Sync
    calls run it directly, as before.
    """

    raise_error = False   # a metering failure must never break a chat

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            usage = _extract_usage(response)
            if usage is None:
                log.warning("metering: %s reply carried no token usage; not recorded", self.model)
                return
            record(self.provider, self.model, *usage)
        except Exception:
            log.warning("metering: could not record usage", exc_info=True)


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
    reserved: int = 0   # tokens this turn holds until release()


# Tokens held by capped clients' turns in flight, and how many of those turns
# have ended. In memory, like chat sessions.
_reserve_lock = threading.Lock()
_reserved: dict[str, int] = {}
_ended: dict[str, int] = {}


def _usage_and_limit(client_id: str) -> tuple[int, int | None]:
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
        return used, None
    return used, _plan_limit(row["plan"])


def check_allowance(client_id: str, reserve: bool = False) -> Allowance:
    """Tokens used this UTC month vs the client's plan. Fails open on any error.

    reserve=True is for a chat turn about to start. Usage is only written as each
    model call ends, so parallel turns would all pass on the same "used" figure;
    instead, tokens held by the client's turns in flight count as used, and an
    allowed turn holds config.TURN_TOKEN_RESERVE more until release(). Uncapped
    clients hold nothing.
    """
    try:
        if not reserve:
            used, limit = _usage_and_limit(client_id)
            return Allowance(limit is None or used < limit, used, limit)
        for _ in range(5):
            with _reserve_lock:
                ended = _ended.get(client_id, 0)
            # Read outside the lock, so release() (on the event loop) never waits on the ledger.
            used, limit = _usage_and_limit(client_id)
            if limit is None:
                return Allowance(True, used, None)
            with _reserve_lock:
                if _ended.get(client_id, 0) != ended:
                    # A turn ended during the read: its hold is gone, but its
                    # usage may be missing from `used`. Read again.
                    continue
                held = _reserved.get(client_id, 0)
                if used + held >= limit:
                    return Allowance(False, used, limit)
                hold = config.TURN_TOKEN_RESERVE
                _reserved[client_id] = held + hold
                return Allowance(True, used, limit, reserved=hold)
        # Turns keep ending under us; this one can try again in a moment.
        return Allowance(False, used, limit)
    except Exception:
        log.error("metering: allowance check failed; allowing the turn", exc_info=True)
        return Allowance(True, 0, None)


def release(client_id: str, allowance: Allowance) -> None:
    """Give back what a turn's check reserved. Call once, when the turn ends
    (after its usage is written)."""
    if not allowance.reserved:
        return
    with _reserve_lock:
        left = _reserved.get(client_id, 0) - allowance.reserved
        if left > 0:
            _reserved[client_id] = left
        else:
            _reserved.pop(client_id, None)
        _ended[client_id] = _ended.get(client_id, 0) + 1


# --- Report ---------------------------------------------------------------

_REPORT_NOTE = (
    "Token counts are exact, as reported by the model provider. Costs are estimates "
    "from pricing.json (prompt-cache discounts ignored); null means the model has no price entry."
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
            "SUM(cost_usd IS NULL) AS unpriced_calls "
            f"FROM usage WHERE {where} GROUP BY client_id, feature, model, day "
            "ORDER BY day, client_id, feature, model", args)]
        totals = {r["client_id"]: dict(r) for r in conn.execute(
            "SELECT client_id, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, "
            f"SUM(cost_usd IS NULL) AS unpriced_calls FROM usage WHERE {where} GROUP BY client_id", args)}
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
            # null only when unpriced calls leave the cost unknown; no calls at all is $0.
            "cost_usd": t.get("cost_usd") if unpriced else (t.get("cost_usd") or 0.0),
            "unpriced_calls": unpriced,
        })
    return {
        "month": label, "timezone": "UTC",
        "period_start_utc": start, "period_end_utc": end,
        "note": _REPORT_NOTE,
        "clients": clients,
        "breakdown": breakdown,
    }
