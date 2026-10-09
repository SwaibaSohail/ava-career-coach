"""App settings, loaded from the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()  # do not override real environment variables (safer in CI/containers)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Guardrails: a small, cheap model for the ingress LLM safety check (stage 5),
# kept separate from the main GROQ_MODEL. Fail-open if it errors (logged).
GUARD_MODEL = os.getenv("GUARD_MODEL", "openai/gpt-oss-20b")
# Reasoning effort for the guard (gpt-oss models): "low" keeps it fast and cuts
# output tokens. Leave it empty for a non-reasoning model so it is not sent.
GUARD_REASONING_EFFORT = os.getenv("GUARD_REASONING_EFFORT", "low")
GUARD_LLM_ENABLED = os.getenv("GUARD_LLM_ENABLED", "true").lower() in ("1", "true", "yes", "on")
# The guard's own call budget (seconds per attempt, retries after the first):
# it fails open, so it gives up well before the main chat's 60 s x 4 retries.
# Retries come half a second apart, never after the wait a rate limit asks for.
GUARD_TIMEOUT_S = float(os.getenv("GUARD_TIMEOUT_S", "8"))
GUARD_MAX_RETRIES = int(os.getenv("GUARD_MAX_RETRIES", "1"))

# Max characters kept from an incoming chat message before it reaches the agent.
# Generous enough to fit a pasted job description; still caps absurd payloads.
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "8000"))

# Reject uploaded files larger than this (bytes) before parsing. Default 10 MB.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# SMTP (outbound email). Port 587 + STARTTLS only; secrets stay in .env.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "") or SMTP_USER

# MCP: servers config file (resolved against backend/) and a cap on how much
# text an MCP tool may feed back to the model.
MCP_SERVERS_FILE = os.getenv("MCP_SERVERS_FILE", "mcp_servers.json")
MCP_MAX_OUTPUT_CHARS = int(os.getenv("MCP_MAX_OUTPUT_CHARS", "4000"))

# Mock interview: how many questions to ask (default from env, clamped to MIN..MAX).
INTERVIEW_DEFAULT_QUESTIONS = int(os.getenv("INTERVIEW_DEFAULT_QUESTIONS", "5"))
INTERVIEW_MIN_QUESTIONS = 3
INTERVIEW_MAX_QUESTIONS = 10

# Local embedding model — runs on-device, no API key or cost.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Token usage metering (internal only — never shown to users). The ledger path
# is resolved against backend/. Plans and model prices are editable JSON files;
# restart the backend after editing them.
METERING_DB = os.getenv("METERING_DB", "data/usage.db")
PLANS_FILE = "plans.json"
PRICING_FILE = "pricing.json"

# Admin usage report (GET /api/admin/usage): closed (404) unless this is set.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# When true, POST /api/session must carry a valid X-Client-Key header; when
# false, chats without a key count under the uncapped built-in "default" client.
REQUIRE_CLIENT_KEY = os.getenv("REQUIRE_CLIENT_KEY", "false").lower() in ("1", "true", "yes", "on")

# Cloudflare Workers AI, for Clef (calibrated classification; see clef.py).
# Nothing calls it unless both are set. A Clef call gets one attempt, no
# retries, with this many seconds to answer (the guard: for all its windows).
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLEF_TIMEOUT_S = float(os.getenv("CLEF_TIMEOUT_S", "3"))

# Who runs the ingress guard's stage 5: "groq" (default), "clef" (Clef decides;
# on a failure CLEF_FALLBACK runs) or "shadow" (Groq decides, Clef is asked in
# the background and only logged). clef/shadow without the Cloudflare keys run
# as groq. See guardrails.py and the README's "Clef guard" section.
GUARD_BACKEND = os.getenv("GUARD_BACKEND", "groq").lower()
CLEF_GUARD_MODEL = os.getenv("CLEF_GUARD_MODEL", "clef")             # clef | clef-flash
CLEF_GUARD_RULE = os.getenv("CLEF_GUARD_RULE", "choice").lower()     # choice | noul
# Tuned on the dev set for question set guard-v1 (choice rule) and held on the
# holdout; see docs/superpowers/specs/2026-10-09-clef-guard-eval-results.md.
# A new question-set version needs a new sweep before this is trusted.
CLEF_BLOCK_THRESHOLD = float(os.getenv("CLEF_BLOCK_THRESHOLD", "0.4"))
# Longer messages can be read as two windows: the first and last this many
# characters. 0 (the default) reads every message as one window: the cutoff
# probe showed Clef reading a 10,000-character message in full, in English and
# Urdu script, and Ava caps messages at 8,000. Kept as a switch in case
# Workers AI starts truncating the state again.
CLEF_WINDOW_CHARS = int(os.getenv("CLEF_WINDOW_CHARS", "0"))
# When Clef fails: "groq" runs the Groq guard on this budget with no retries;
# "open" skips the check (the regex and rule stages still apply).
CLEF_FALLBACK = os.getenv("CLEF_FALLBACK", "groq").lower()
CLEF_FALLBACK_TIMEOUT_S = float(os.getenv("CLEF_FALLBACK_TIMEOUT_S", "5"))
# Decision log for Clef guard checks (resolved against backend/; never message
# text). Shadow mode may keep the text of messages Clef and Groq disagree on,
# for this many days, only while GUARD_SHADOW_STORE_TEXT is on (dev only).
GUARD_LOG_DB = os.getenv("GUARD_LOG_DB", "data/guard.db")
GUARD_SHADOW_STORE_TEXT = os.getenv("GUARD_SHADOW_STORE_TEXT", "false").lower() in ("1", "true", "yes", "on")
GUARD_SHADOW_RETENTION_DAYS = int(os.getenv("GUARD_SHADOW_RETENTION_DAYS", "14"))


def is_api_key_configured() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "your_groq_api_key_here"


def is_tavily_configured() -> bool:
    return bool(TAVILY_API_KEY) and TAVILY_API_KEY != "your_tavily_api_key_here"


def is_smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)
