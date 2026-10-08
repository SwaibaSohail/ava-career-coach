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


def is_api_key_configured() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "your_groq_api_key_here"


def is_tavily_configured() -> bool:
    return bool(TAVILY_API_KEY) and TAVILY_API_KEY != "your_tavily_api_key_here"


def is_smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)
