"""The ingress guard's decision log: a small SQLite file at GUARD_LOG_DB
(default backend/data/guard.db, gitignored).

Every Clef guard check, in clef or shadow mode, writes one row: the question-set
version, rule, threshold, model, number of windows, Clef's seven probabilities,
the labels, the time taken and the error kind, if any. Never the message, so the
threshold can be re-tuned from real traffic without keeping what people wrote.

The only text kept is shadow mode's, and only with GUARD_SHADOW_STORE_TEXT on (a
dev flag): messages where Clef and Groq disagreed, deleted after
GUARD_SHADOW_RETENTION_DAYS (the backend purges at startup and every hour, and
before listing them) and overwritten on disk, not just unlinked.

Writes are queued for one writer thread and never raise into the guard: a
failure is logged as a warning and the row is dropped.
"""

import atexit
import logging
import os
import queue
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone

import config

log = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUSY_TIMEOUT_S = 2   # how long the writer (or a purge) waits on a locked log before giving up

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
    groq_label     TEXT,              -- shadow mode: Groq's label; NULL when Groq failed (the user got CLEAN)
    latency_ms     REAL,
    error          TEXT               -- the ClefError kind (or "unexpected"), when Clef failed
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
    # The year padded by hand: %Y may not pad years before 1000, and the log
    # compares these as text.
    return f"{dt.year:04d}" + dt.strftime("-%m-%dT%H:%M:%S.%fZ")


def _db_path() -> str:
    path = config.GUARD_LOG_DB
    return path if os.path.isabs(path) else os.path.join(_BASE_DIR, path)


def _connect(path: str) -> sqlite3.Connection:
    """A fresh connection per call (safe across threads), schema ensured once per path."""
    with _init_lock:
        if path not in _initialized:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with closing(sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
            _initialized.add(path)
    conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    # Deleted rows are overwritten, not left readable in free pages.
    conn.execute("PRAGMA secure_delete=ON")
    return conn


def _insert(path: str, table: str, row: dict) -> None:
    with closing(_connect(path)) as conn, conn:
        conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                     list(row.values()))


# Rows go through one writer thread: a clef-mode check logs its decision before
# the reply starts, and must never wait on SQLite for it (a locked log would hold
# up the reply for the busy timeout); one writer also means our own rows never
# contend for the write lock.
_pending: queue.Queue = queue.Queue()
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()
_NOUNS = {"guard_decisions": "decision", "guard_disagreements": "disagreement"}


def _write_pending() -> None:
    while True:
        path, table, row = _pending.get()
        try:
            _insert(path, table, row)
        except Exception as exc:
            log.warning("guard log: could not record a %s (%s: %s)", _NOUNS[table], type(exc).__name__, exc)
        finally:
            _pending.task_done()


def _enqueue(table: str, row: dict) -> None:
    """Queue one row for the writer, for the log GUARD_LOG_DB names now. Returns at once."""
    global _writer
    with _writer_lock:
        if _writer is None:
            _writer = threading.Thread(target=_write_pending, name="guard-log-writer", daemon=True)
            _writer.start()
    _pending.put((_db_path(), table, row))


def flush() -> None:
    """Block until every queued row is written (or has failed and been logged)."""
    _pending.join()


atexit.register(flush)


def _probability_columns(probabilities: dict | None) -> dict:
    return {f"p_{name}": (probabilities or {}).get(name) for name in PROBABILITIES}


def record_decision(*, mode, qset_version, rule, threshold, model, windows, probabilities=None,
                    clef_label=None, groq_label=None, latency_ms=None, error=None) -> None:
    """Log one Clef guard check (no text). Queued; never raises."""
    try:
        _enqueue("guard_decisions", {
            "ts_utc": _iso(_utcnow()), "mode": mode, "qset_version": qset_version, "rule": rule,
            "threshold": threshold, "model": model, "windows": windows, **_probability_columns(probabilities),
            "clef_label": clef_label, "groq_label": groq_label, "latency_ms": latency_ms, "error": error,
        })
    except Exception as exc:
        log.warning("guard log: could not record a decision (%s: %s)", type(exc).__name__, exc)


def record_disagreement(*, qset_version, model, groq_label, clef_label, probabilities, message) -> None:
    """Keep a message Clef and Groq labelled differently, with its text (shadow
    mode with GUARD_SHADOW_STORE_TEXT on). Queued; never raises."""
    try:
        _enqueue("guard_disagreements", {
            "ts_utc": _iso(_utcnow()), "qset_version": qset_version, "model": model,
            "groq_label": groq_label, "clef_label": clef_label, **_probability_columns(probabilities),
            "message": message,
        })
    except Exception as exc:
        log.warning("guard log: could not record a disagreement (%s: %s)", type(exc).__name__, exc)


def recent_disagreements(limit: int = 50) -> list[dict]:
    """The stored disagreements still inside the retention period, newest first,
    text included."""
    flush()
    purge_old()
    with closing(_connect(_db_path())) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM guard_disagreements ORDER BY id DESC LIMIT ?", (limit,))]


def purge_old() -> int:
    """Delete disagreements older than GUARD_SHADOW_RETENTION_DAYS; returns how
    many went. Never raises."""
    path = _db_path()
    if not os.path.exists(path):
        return 0   # nothing was ever logged
    try:
        cutoff = _iso(_utcnow() - timedelta(days=config.GUARD_SHADOW_RETENTION_DAYS))
    except OverflowError:
        return 0   # a retention reaching back past year 1 ("for ever"): nothing is old enough
    try:
        with closing(_connect(path)) as conn:
            with conn:
                gone = conn.execute("DELETE FROM guard_disagreements WHERE ts_utc < ?", (cutoff,)).rowcount
            if gone:
                # secure_delete zeroed the freed pages in the WAL; this writes them
                # over the database file and empties the WAL, which still held the text.
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return gone
    except Exception as exc:
        log.warning("guard log: could not purge old disagreements (%s: %s)", type(exc).__name__, exc)
        return 0
