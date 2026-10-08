"""Factory for the Groq chat model, shared by the analyzer, chat and agent.

Every model gets a metering.UsageRecorder, so all token usage is recorded here,
in one place, for every feature.
"""

from langchain_groq import ChatGroq

import config
import metering


def get_llm(temperature: float = 0.2, model: str | None = None,
            reasoning_effort: str | None = None, timeout: float = 60,
            max_retries: int = 4) -> ChatGroq:
    name = model or config.GROQ_MODEL
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=name,
        temperature=temperature,
        # Reasoning models only; empty means leave it unset (ChatGroq's default).
        reasoning_effort=reasoning_effort or None,
        # Retry brief network blips rather than surfacing them to the user.
        # The guard passes a tighter budget: it fails open instead of waiting.
        max_retries=max_retries,
        timeout=timeout,
        # Constructor callbacks only (never also at invoke time): one row per call.
        callbacks=[metering.UsageRecorder("groq", name)],
    )
