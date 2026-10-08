"""The ingress guard's decision log: a small SQLite file at GUARD_LOG_DB
(default backend/data/guard.db, gitignored).

Every Clef guard check, in clef or shadow mode, writes one row: the question-set
version, rule, threshold, model, number of windows, Clef's seven probabilities,
the labels, the time taken and the error kind, if any. Never the message, so the
threshold can be re-tuned from real traffic without keeping what people wrote.

The only text kept is shadow mode's, and only with GUARD_SHADOW_STORE_TEXT on (a
dev flag): messages where Clef and Groq disagreed, deleted after
GUARD_SHADOW_RETENTION_DAYS.

Writes never raise into the guard: a failure is logged as a warning and the row
is dropped.
"""

import logging
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone

import config

log = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUSY_TIMEOUT_S = 2   # a locked log must not hold up the guard for long

# Clef's answers for one message: the choice over the four classes, then the
# three yes/no questions. Stored as p_<name>.
PROBABILITIES = ("clean", "injection", "abuse", "harmful", "is_injection", "is_abuse", "is_harmful")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS guard_decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT NOT NULL,
    mode           TEXT NOT NULL,     -- clef | shadow
    qset_version   TEXT NOT NULL,
    rule           TEXT NOT NULL,
    threshold      REAL NOT NULL,
    model          TEXT NOT NULL,
    windows        INTEGER NOT NULL,
    p_clean        REAL,              -- the probabilities: NULL when Clef failed
    p_injection    REAL,
    p_abuse        REAL,
    p_harmful      REAL,
    p_is_injection REAL,
    p_is_abuse     REAL,
    p_is_harmful   REAL,
    clef_label     TEXT,
    groq_label     TEXT,              -- shadow mode: the label the user got
    latency_ms     REAL,
    error          TEXT               -- the ClefError kind, when Clef failed
);
CREATE TABLE IF NOT EXISTS guard_disagreements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT NOT NULL,
    qset_version   TEXT NOT NULL,
    model          TEXT NOT NULL,
    groq_label     TEXT NOT NULL,
    clef_label     TEXT NOT NULL,
    p_clean        REAL,
    p_injection    REAL,
    p_abuse        REAL,
    p_harmful      REAL,
    p_is_injection REAL,
    p_is_abuse     REAL,
    p_is_harmful   REAL,
    message        TEXT NOT NULL
);
"""

_init_lock = threading.Lock()
_initialized: set[str] = set()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _db_path() -> str:
    path = config.GUARD_LOG_DB
    return path if os.path.isabs(path) else os.path.join(_BASE_DIR, path)


def _connect() -> sqlite3.Connection:
    """A fresh connection per call (safe across threads), schema ensured once per path."""
    path = _db_path()
    with _init_lock:
        if path not in _initialized:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with closing(sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
            _initialized.add(path)
    conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    return conn


def _insert(table: str, row: dict) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                     list(row.values()))


def _probability_columns(probabilities: dict | None) -> dict:
    return {f"p_{name}": (probabilities or {}).get(name) for name in PROBABILITIES}


def record_decision(*, mode, qset_version, rule, threshold, model, windows, probabilities=None,
                    clef_label=None, groq_label=None, latency_ms=None, error=None) -> None:
    """Log one Clef guard check (no text). Never raises."""
    try:
        _insert("guard_decisions", {
            "ts_utc": _iso(_utcnow()), "mode": mode, "qset_version": qset_version, "rule": rule,
            "threshold": threshold, "model": model, "windows": windows, **_probability_columns(probabilities),
            "clef_label": clef_label, "groq_label": groq_label, "latency_ms": latency_ms, "error": error,
        })
    except Exception as exc:
        log.warning("guard log: could not record a decision (%s: %s)", type(exc).__name__, exc)


def record_disagreement(*, qset_version, model, groq_label, clef_label, probabilities, message) -> None:
    """Keep a message Clef and Groq labelled differently, with its text (shadow
    mode with GUARD_SHADOW_STORE_TEXT on). Never raises."""
    try:
        _insert("guard_disagreements", {
            "ts_utc": _iso(_utcnow()), "qset_version": qset_version, "model": model,
            "groq_label": groq_label, "clef_label": clef_label, **_probability_columns(probabilities),
            "message": message,
        })
    except Exception as exc:
        log.warning("guard log: could not record a disagreement (%s: %s)", type(exc).__name__, exc)


def recent_disagreements(limit: int = 50) -> list[dict]:
    """The stored disagreements, newest first, text included."""
    with closing(_connect()) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM guard_disagreements ORDER BY id DESC LIMIT ?", (limit,))]


def purge_old() -> int:
    """Delete disagreements older than GUARD_SHADOW_RETENTION_DAYS; returns how
    many went. Never raises."""
    cutoff = _iso(_utcnow() - timedelta(days=config.GUARD_SHADOW_RETENTION_DAYS))
    try:
        with closing(_connect()) as conn, conn:
            return conn.execute("DELETE FROM guard_disagreements WHERE ts_utc < ?", (cutoff,)).rowcount
    except Exception as exc:
        log.warning("guard log: could not purge old disagreements (%s: %s)", type(exc).__name__, exc)
        return 0
