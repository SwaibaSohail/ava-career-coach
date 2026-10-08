"""Cloudflare Clef: calibrated answers about a piece of text.

A small client for the Workers AI REST API that knows nothing about its
callers. You pass a state (the text, or any JSON) and named questions: noul
(P(yes)), choice (one of N options) or score (a level on a scale). You get back
probabilities. Thresholds and fallbacks belong to the caller.

One attempt per call, no retries, and a hard timeout (CLEF_TIMEOUT_S), so a
slow Clef costs the caller seconds, not minutes. A failure raises ClefError
with a kind the caller can act on. Each failed request is logged once with the
model, kind, HTTP status, Cloudflare code, request id and time. The state and
the token are never logged.

The free 10,000 Neurons a day are shared by all Workers AI use on the account.
Once they run out (HTTP 429, code 3036), Clef is not called again until they
reset at 00:00 UTC (05:00 in Pakistan). That is logged once at ERROR, so a used-up
allowance never reads as a flaky Clef.

Successful calls are metered (provider "cloudflare", the Workers AI model id,
feature from the caller's context). Failed calls are not, as with Groq.
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

import config
import metering

log = logging.getLogger(__name__)

MODELS = {"clef": "@cf/cloudflare/clef", "clef-flash": "@cf/cloudflare/clef-flash"}
URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model_id}"

_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_MAX_QUESTIONS = 64
_CHOICE_OPTIONS = (2, 255)
_SCORE_LEVELS = (2, 10)
_SUM_TOLERANCE = 0.02   # probabilities come rounded to 4 decimals

_QUOTA_CODE, _CAPACITY_CODE = 3036, 3040


# --- Questions and answers --------------------------------------------------

@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: dict | None = None    # optional {"true": ..., "false": ...}


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: dict[str, str]        # option id -> description, 2-255 options


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list[str]             # 2-10 levels, lowest first


@dataclass(frozen=True)
class NoulAnswer:
    noul: float                     # P(yes)


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str                     # the most likely option
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScoreAnswer:
    score: float                    # probability-weighted level, 0-based
    probabilities: dict[int, float]
    legend: dict[int, str]


@dataclass(frozen=True)
class Decision:
    model: str            # Workers AI id called, e.g. "@cf/cloudflare/clef" (= pricing key)
    answers: dict         # id -> NoulAnswer | ChoiceAnswer | ScoreAnswer
    input_tokens: int
    output_tokens: int
    latency_ms: float
    request_id: str | None


_TYPES = {Noul: "noul", Choice: "choice", Score: "score"}


class ClefError(RuntimeError):
    """A Clef call that failed or was refused. Never carries the state or the token.

    kind: config (no credentials), auth, quota (the free daily allowance is used
    up), capacity (Clef is busy), bad_request, timeout, server (5xx or the
    network) or protocol (an answer we can't read). status and code are the
    HTTP status and Cloudflare's error code, when there was a response."""

    def __init__(self, kind, message, *, status=None, code=None):
        super().__init__(message)
        self.kind, self.status, self.code = kind, status, code


def is_configured() -> bool:
    return bool(config.CLOUDFLARE_ACCOUNT_ID and config.CLOUDFLARE_API_TOKEN)


# --- Request ----------------------------------------------------------------

def _bad_request(message):
    return ClefError("bad_request", message)


def _question(qid, question) -> dict:
    kind = _TYPES.get(type(question))
    if kind is None:
        raise _bad_request(f"question {qid!r} is not a Noul, Choice or Score")
    if not question.instructions:
        raise _bad_request(f"question {qid!r} has no instructions")
    body = {"type": kind, "instructions": question.instructions}
    if kind == "choice":
        low, high = _CHOICE_OPTIONS
        if not isinstance(question.criteria, dict) or not low <= len(question.criteria) <= high:
            raise _bad_request(f"choice {qid!r} needs {low}-{high} options")
        body["criteria"] = dict(question.criteria)
    elif kind == "score":
        low, high = _SCORE_LEVELS
        if not isinstance(question.criteria, (list, tuple)) or not low <= len(question.criteria) <= high:
            raise _bad_request(f"score {qid!r} needs {low}-{high} levels")
        body["criteria"] = list(question.criteria)
    elif question.criteria is not None:
        body["criteria"] = dict(question.criteria)
    return body


def build_body(model: str, state, questions: dict) -> dict:
    """The request body: exactly model, state and questions (the hosted API may
    reject anything else). Checks Clef's limits first; raises ClefError("bad_request")."""
    if model not in MODELS:
        raise _bad_request(f"unknown Clef model {model!r}; use one of {', '.join(MODELS)}")
    if not 1 <= len(questions) <= _MAX_QUESTIONS:
        raise _bad_request(f"a call takes 1-{_MAX_QUESTIONS} questions, not {len(questions)}")
    for qid in questions:
        if not isinstance(qid, str) or not _ID_RE.fullmatch(qid):
            raise _bad_request(f"question id {qid!r} must be 1-100 letters, digits, '_', '.' or '-'")
    return {"model": model, "state": state,
            "questions": {qid: _question(qid, q) for qid, q in questions.items()}}


# --- Response ---------------------------------------------------------------

def _protocol(message):
    return ClefError("protocol", message)


def _kind(status: int | None, code: int | None) -> str:
    """Cloudflare's code first (it tells a used-up allowance from a busy model), then the status."""
    if code == _QUOTA_CODE:
        return "quota"
    if code == _CAPACITY_CODE or status == 429:
        return "capacity"
    if status in (401, 403):
        return "auth"
    if status == 408:
        return "timeout"
    if status is not None and 400 <= status < 500:
        return "bad_request"
    if status is not None and status >= 500:
        return "server"
    return "protocol"


def _error_code(payload) -> int | None:
    """The first Cloudflare error code in a v4 envelope, if any."""
    errors = payload.get("errors") if isinstance(payload, dict) else None
    for item in errors if isinstance(errors, list) else []:
        if isinstance(item, dict) and isinstance(item.get("code"), int):
            return item["code"]
    return None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _probability(qid, value) -> float:
    if not _is_number(value) or not 0 <= value <= 1:
        raise _protocol(f"answer {qid!r} has a value that is not a probability")
    return float(value)


def _distribution(qid, probabilities, options) -> dict:
    """Probabilities for exactly these options, summing to 1 (give or take rounding)."""
    if not isinstance(probabilities, dict) or set(probabilities) != set(options):
        raise _protocol(f"answer {qid!r} does not give one probability per option")
    values = {option: _probability(qid, p) for option, p in probabilities.items()}
    if abs(sum(values.values()) - 1) > _SUM_TOLERANCE:
        raise _protocol(f"answer {qid!r} has probabilities summing to {sum(values.values()):.3f}")
    return values


def _by_level(qid, mapping) -> dict:
    """A score's {"0": ..., "1": ...} keyed by int."""
    try:
        return {int(level): value for level, value in mapping.items()}
    except (AttributeError, TypeError, ValueError):
        raise _protocol(f"answer {qid!r} has levels that are not numbers") from None


def _answer(qid, question, raw):
    if isinstance(question, Noul):
        return NoulAnswer(_probability(qid, raw.get("noul")))
    if isinstance(question, Choice):
        probabilities = _distribution(qid, raw.get("probabilities"), question.criteria)
        if not isinstance(raw.get("choice"), str) or raw["choice"] not in question.criteria:
            raise _protocol(f"answer {qid!r} chose something that is not an option")
        return ChoiceAnswer(raw["choice"], probabilities)
    if "legend" not in raw or not _is_number(raw.get("score")):
        raise _protocol(f"answer {qid!r} has no score or legend")
    probabilities = _distribution(qid, _by_level(qid, raw.get("probabilities")), range(len(question.criteria)))
    return ScoreAnswer(float(raw["score"]), probabilities, _by_level(qid, raw["legend"]))


def parse(payload: dict, questions: dict) -> tuple[dict, int, int]:
    """(answers by question id, input tokens, output tokens) from a response
    body, in Cloudflare's v4 envelope or bare. Raises ClefError for a failure
    envelope, and kind "protocol" for anything malformed."""
    if not isinstance(payload, dict):
        raise _protocol("the response is not a JSON object")
    if payload.get("success") is False:
        code = _error_code(payload)
        raise ClefError(_kind(None, code), f"Cloudflare reported a failure (code {code})", code=code)
    result = payload["result"] if "result" in payload else payload
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise _protocol("the response has no answers")
    usage = result.get("usage")
    tokens = (usage.get("input_tokens"), usage.get("output_tokens")) if isinstance(usage, dict) else (None, None)
    if not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in tokens):
        raise _protocol("the response has no token usage")
    answers = {}
    for qid, question in questions.items():
        raw = result["answers"].get(qid)
        if not isinstance(raw, dict) or raw.get("type") != _TYPES.get(type(question)):
            raise _protocol(f"no {_TYPES.get(type(question))} answer for {qid!r}")
        answers[qid] = _answer(qid, question, raw)
    return answers, tokens[0], tokens[1]


# --- Daily allowance --------------------------------------------------------

_quota_lock = threading.Lock()
_quota_until: datetime | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def quota_exhausted_until() -> datetime | None:
    """When the used-up free allowance resets (the next 00:00 UTC), or None
    while Clef may be called."""
    global _quota_until
    with _quota_lock:
        if _quota_until is not None and _utcnow() >= _quota_until:
            _quota_until = None
        return _quota_until


def _pause_for_quota(model_id, error, request_id) -> None:
    """Stop calling Clef until the allowance resets; logged by the first caller only."""
    global _quota_until
    now = _utcnow()
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    with _quota_lock:
        if _quota_until is not None and now < _quota_until:
            return
        _quota_until = reset
    log.error("clef: the account's free daily Workers AI allowance is used up (%s, HTTP %s, code %s, "
              "request %s); not calling Clef until it resets at %s UTC (05:00 PKT)",
              model_id, error.status, error.code, request_id, f"{reset:%Y-%m-%d %H:%M}")


# --- Calls ------------------------------------------------------------------

_transport: httpx.BaseTransport | None = None   # tests install an httpx.MockTransport here
_clients_lock = threading.Lock()
_sync_client: tuple | None = None    # (transport, httpx.Client)
_async_client: tuple | None = None   # (transport, event loop, httpx.AsyncClient)


def _client() -> httpx.Client:
    """One shared client, so calls reuse connections; rebuilt if _transport changes."""
    global _sync_client
    with _clients_lock:
        if _sync_client is None or _sync_client[0] is not _transport:
            _sync_client = (_transport, httpx.Client(transport=_transport))
        return _sync_client[1]


def _aclient() -> httpx.AsyncClient:
    """As _client, per event loop: pooled connections belong to the loop that opened them."""
    global _async_client
    loop = asyncio.get_running_loop()
    with _clients_lock:
        if _async_client is None or _async_client[0] is not _transport or _async_client[1] is not loop:
            _async_client = (_transport, loop, httpx.AsyncClient(transport=_transport))
        return _async_client[2]


def _prepare(model, state, questions) -> tuple[str, dict, dict, float]:
    """(url, body, headers, started) for one call, or a ClefError refusing it
    without a request."""
    body = build_body(model, state, questions)
    if not is_configured():
        raise ClefError("config", "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must both be set")
    until = quota_exhausted_until()
    if until is not None:
        raise ClefError("quota", f"the free daily allowance is used up; Clef is paused until "
                                 f"{until:%Y-%m-%d %H:%M} UTC")
    url = URL.format(account=config.CLOUDFLARE_ACCOUNT_ID, model_id=MODELS[model])
    headers = {"Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}"}
    return url, body, headers, time.perf_counter()


def _failed(model, error, started, request_id=None) -> ClefError:
    """Log one failed request (never the state or the token) and return its error."""
    if error.kind == "quota":
        _pause_for_quota(MODELS[model], error, request_id)
    else:
        log.warning("clef: %s failed after %.0f ms (%s: %s; request %s)",
                    MODELS[model], (time.perf_counter() - started) * 1000, error.kind, error, request_id)
    return error


def _transport_failed(model, exc, started) -> ClefError:
    kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "server"
    return _failed(model, ClefError(kind, type(exc).__name__), started)


def _finish(model, questions, response, started, feature) -> Decision:
    """The Decision from a response, metered; or its ClefError, logged."""
    latency_ms = (time.perf_counter() - started) * 1000
    request_id = response.headers.get("cf-ai-req-id")
    try:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not response.is_success:
            # Only the status and code: Cloudflare's message may quote the input.
            code = _error_code(payload)
            raise ClefError(_kind(response.status_code, code), f"HTTP {response.status_code}, code {code}",
                            status=response.status_code, code=code)
        if payload is None:
            raise _protocol(f"HTTP {response.status_code} with a body that is not JSON")
        answers, input_tokens, output_tokens = parse(payload, questions)
    except ClefError as error:
        raise _failed(model, error, started, request_id) from None
    decision = Decision(MODELS[model], answers, input_tokens, output_tokens, latency_ms, request_id)
    with metering.feature(feature):
        metering.enqueue("cloudflare", decision.model, input_tokens, output_tokens)
    return decision


def decide(state, questions: dict, *, model: str = "clef", feature: str | None = None,
           timeout: float | None = None) -> Decision:
    """Ask Clef the questions about the state. Blocking: call it off the event
    loop. feature, when given, attributes the call in the ledger; timeout
    defaults to CLEF_TIMEOUT_S. Raises ClefError."""
    url, body, headers, started = _prepare(model, state, questions)
    try:
        response = _client().post(url, json=body, headers=headers, timeout=timeout or config.CLEF_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise _transport_failed(model, exc, started) from exc
    return _finish(model, questions, response, started, feature)


async def adecide(state, questions: dict, *, model: str = "clef", feature: str | None = None,
                  timeout: float | None = None) -> Decision:
    """decide, for async callers."""
    url, body, headers, started = _prepare(model, state, questions)
    try:
        response = await _aclient().post(url, json=body, headers=headers, timeout=timeout or config.CLEF_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise _transport_failed(model, exc, started) from exc
    return _finish(model, questions, response, started, feature)
