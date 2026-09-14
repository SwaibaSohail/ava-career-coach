import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import config
import mcp_client


def test_load_servers_reads_file(tmp_path, monkeypatch):
    f = tmp_path / "servers.json"
    f.write_text(json.dumps({"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}))
    monkeypatch.setattr(mcp_client, "_servers_path", lambda: str(f))
    assert mcp_client.is_mcp_configured() is True
    assert "fetch" in mcp_client._load_servers()


def test_load_servers_missing_file(monkeypatch):
    monkeypatch.setattr(mcp_client, "_servers_path", lambda: "/no/such/file.json")
    assert mcp_client._load_servers() == {}
    assert mcp_client.is_mcp_configured() is False


def test_load_servers_invalid_json(tmp_path, monkeypatch):
    f = tmp_path / "bad.json"
    f.write_text("{ not json")
    monkeypatch.setattr(mcp_client, "_servers_path", lambda: str(f))
    assert mcp_client._load_servers() == {}
