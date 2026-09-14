"""MCP client: load tools from configured MCP servers into Ava.

Config-driven and fail-safe — if no server is configured or a connection fails,
this yields no tools and the app runs unchanged. MCP tools are async-only (they
expose `coroutine`), so they can only be called from async code (Ava's astream /
chat_with_ava's asyncio.run), never synchronously.
"""

import asyncio
import json
import logging
import os

from langchain_core.tools import StructuredTool

import config
from cv_processor import sanitize_cv_text

log = logging.getLogger(__name__)

_TRUNC = "\n…[truncated]"


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


def _clamp_output(text: str) -> str:
    """Treat MCP output as untrusted data: scrub injection lines, then cap size."""
    cleaned = sanitize_cv_text(text or "")
    if len(cleaned) > config.MCP_MAX_OUTPUT_CHARS:
        cleaned = cleaned[: config.MCP_MAX_OUTPUT_CHARS] + _TRUNC
    return cleaned


def _guard_tool(tool):
    """Wrap an MCP tool so its string result is sanitized + capped."""
    async def _run(**kwargs):
        result = await tool.ainvoke(kwargs)
        return _clamp_output(result) if isinstance(result, str) else result

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


_BUILTIN_TOOL_NAMES = {
    "web_search", "job_search", "search_cv", "save_document", "propose_email",
}
_cache: list = []


async def _fetch_tools(servers: dict) -> list:
    """Connect to the configured servers and return their raw tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(servers)
    return await client.get_tools()


async def load_mcp_tools_async() -> list:
    servers = _load_servers()
    if not servers:
        return []
    try:
        raw = await asyncio.wait_for(_fetch_tools(servers), timeout=20)
    except Exception as exc:
        log.warning("MCP disabled (could not load tools): %s", exc)
        return []
    tools = []
    for t in raw:
        if t.name in _BUILTIN_TOOL_NAMES:
            log.warning("MCP: dropping tool %r (collides with a built-in)", t.name)
            continue
        tools.append(_guard_tool(t))
    log.info("MCP: loaded %d tool(s) from %d server(s)", len(tools), len(servers))
    return tools


async def init_mcp() -> None:
    global _cache
    _cache = await load_mcp_tools_async()


def get_mcp_tools() -> list:
    return _cache
