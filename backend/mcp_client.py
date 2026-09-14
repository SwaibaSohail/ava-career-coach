"""MCP client: load tools from configured MCP servers into Ava.

Config-driven and fail-safe — if no server is configured or a connection fails,
this yields no tools and the app runs unchanged. MCP tools are async-only (they
expose `coroutine`), so they can only be called from async code (Ava's astream /
chat_with_ava's asyncio.run), never synchronously.
"""

import json
import logging
import os

import config

log = logging.getLogger(__name__)


def _servers_path() -> str:
    # Resolve against this module's dir, NOT the CWD (uvicorn often starts from
    # the repo root, which would make a relative path silently "not configured").
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MCP_SERVERS_FILE)


def _load_servers() -> dict:
    path = _servers_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("MCP: could not read %s: %s", path, exc)
        return {}


def is_mcp_configured() -> bool:
    return bool(_load_servers())
