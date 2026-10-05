"""Factory for the Groq chat model, shared by the analyzer, chat and agent.

Every model gets a metering.UsageRecorder, so all token usage is recorded here,
in one place, for every feature.
"""

from langchain_groq import ChatGroq

import config
import metering


def get_llm(temperature: float = 0.2, model: str | None = None) -> ChatGroq:
    name = model or config.GROQ_MODEL
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=name,
        temperature=temperature,
        # Retry brief network blips rather than surfacing them to the user.
        max_retries=4,
        timeout=60,
        # Constructor callbacks only (never also at invoke time): one row per call.
        callbacks=[metering.UsageRecorder("groq", name)],
    )
