"""Layer 1 ingress guardrails: filter chat messages before the main agent runs.

Stages run cheapest-first so junk is rejected at ~$0. Stages 1-4 are pure
Python; only stage 5 (check_input_llm) makes a paid API call, and it runs only
if 1-4 pass. See docs/superpowers/specs/2026-09-05-ingress-guardrails-design.md.
"""

import unicodedata
from dataclasses import dataclass


@dataclass
class GuardResult:
    allowed: bool
    cleaned_message: str
    category: str | None
    safe_reply: str | None
    dialect: str


# --- Stage 1: length + control-char truncate ---------------------------------

def truncate(message: str, max_chars: int = 500) -> str:
    """Cap length and strip control characters (keep newlines and tabs)."""
    text = message or ""
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )
    return text[:max_chars].strip()


# --- Guarded replies (canned; NEVER calls an LLM) ----------------------------

_REPLIES = {
    "injection": (
        "I can't follow instructions embedded inside a message, but I'm glad to "
        "help with your CV, a cover letter, or finding roles. What would you like "
        "to work on?"
    ),
    "abuse": (
        "I'd like to keep this respectful, so I won't engage with that. I'm here "
        "whenever you want help with your CV or job search."
    ),
    "spam": (
        "That looks like spam, so I'll skip it. Tell me the role you're aiming for "
        "and I'll help with your CV or a cover letter."
    ),
    "harmful": (
        "I can't help with that, but I'm happy to help with your CV, a cover "
        "letter, or your job search."
    ),
}


def generate_guarded_reply(category: str | None, dialect: str = "english") -> str:
    """Return canned, friendly text for a blocked message. No LLM call.

    `dialect` is accepted for future per-dialect phrasing; v1 replies in English.
    """
    return _REPLIES.get(category or "", _REPLIES["harmful"])


# --- Pipeline entry point -----------------------------------------------------

def guard_incoming(message: str) -> GuardResult:
    cleaned = truncate(message)
    dialect = "english"
    return GuardResult(True, cleaned, None, None, dialect)
