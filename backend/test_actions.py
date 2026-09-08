import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import config
import session_store


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    # No live Groq from the ingress guard, and isolate the global session store.
    monkeypatch.setattr(config, "GUARD_LLM_ENABLED", False)
    monkeypatch.setattr(session_store, "_sessions", {})


def test_is_smtp_configured(monkeypatch):
    for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.setattr(config, name, "x")
    assert config.is_smtp_configured() is True
    monkeypatch.setattr(config, "SMTP_PASSWORD", "")
    assert config.is_smtp_configured() is False
