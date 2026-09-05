"""Factory for the Groq chat model, shared by the analyzer, chat and agent."""

from langchain_groq import ChatGroq

import config


def get_llm(temperature: float = 0.2, model: str | None = None) -> ChatGroq:
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=model or config.GROQ_MODEL,
        temperature=temperature,
        # Retry brief network blips rather than surfacing them to the user.
        max_retries=4,
        timeout=60,
    )
