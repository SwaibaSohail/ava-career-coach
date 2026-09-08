"""App settings, loaded from the .env file."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Guardrails: a small, cheap model for the ingress LLM safety check (stage 5),
# kept separate from the main GROQ_MODEL. Fail-open if it errors.
GUARD_MODEL = os.getenv("GUARD_MODEL", "llama-3.1-8b-instant")
GUARD_LLM_ENABLED = os.getenv("GUARD_LLM_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# Max characters kept from an incoming chat message before it reaches the agent.
# Generous enough to fit a pasted job description; still caps absurd payloads.
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "8000"))

# Local embedding model — runs on-device, no API key or cost.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def is_api_key_configured() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "your_groq_api_key_here"


def is_tavily_configured() -> bool:
    return bool(TAVILY_API_KEY) and TAVILY_API_KEY != "your_tavily_api_key_here"
