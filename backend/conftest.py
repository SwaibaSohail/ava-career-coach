import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import config


@pytest.fixture(autouse=True)
def _temp_usage_ledger(tmp_path, monkeypatch):
    """Every test gets its own throwaway usage ledger; nothing touches data/usage.db."""
    monkeypatch.setattr(config, "METERING_DB", str(tmp_path / "usage.db"))
