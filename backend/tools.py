"""Agent tools: web_search and job_search (both via Tavily)."""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

import config


def _tavily(max_results: int = 3) -> TavilySearch:
    """Create a Tavily search tool using the key from .env."""
    return TavilySearch(
        max_results=max_results,
        tavily_api_key=config.TAVILY_API_KEY,
        topic="general",
    )


# Keep each result's snippet short so we don't blow past the model's
# tokens-per-minute limit when all results get fed back into the agent,
# but long enough to keep company name / salary / location detail.
_MAX_SNIPPET_CHARS = 500


def _format_results(raw) -> str:
    """Turn Tavily's JSON into a short, readable list for the agent."""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return str(raw)

    results = raw.get("results") or []
    if not results:
        return "No results found."

    lines = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = (item.get("content") or "").strip()[:_MAX_SNIPPET_CHARS]
        lines.append(f"{i}. {title}\n   URL: {url}\n   {content}")
    return "\n\n".join(lines)


def _tavily_search(query: str, max_results: int, label: str, **extra) -> str:
    """Run a Tavily query and format it, or return a friendly error string."""
    if not config.is_tavily_configured():
        return "Tavily is not configured. Add TAVILY_API_KEY to your .env file."
    try:
        raw = _tavily(max_results=max_results).invoke({"query": query, **extra})
        return _format_results(raw)
    except Exception as exc:
        return f"{label} failed: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the live web for current information (news, companies, skills, salaries).

    Args:
        query: What to search for.
    """
    return _tavily_search(query, 3, "Web search")


@tool
def job_search(role: str, skills: str, location: str = "remote") -> str:
    """Search the web for current job openings that match a role, skills, and location.

    Args:
        role: Job title to look for, e.g. "backend engineer".
        skills: Comma-separated skills from the CV, e.g. "Python, FastAPI, SQL".
        location: City, country, or "remote".
    """
    query = f"{role} jobs {location} hiring {skills}"
    return _tavily_search(query, 5, "Job search", time_range="month")


def get_tools() -> list:
    """Return the list of tools we hand to the agent."""
    return [web_search, job_search]
