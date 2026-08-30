"""Agent tools: calculator (local math), web_search and job_search (both via Tavily)."""

import ast
import operator

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

import config

# Only the math operations we allow. This is safer than Python's eval().
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_math(node):
    """Walk an AST and evaluate only simple arithmetic."""
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_math(node.left), _eval_math(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("Only basic arithmetic is allowed (+ - * / ** %).")


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


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression. Use this for scores, averages, or comparisons.

    Args:
        expression: A math expression such as "85 - 12" or "(3 + 4) / 2".
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_math(tree)
        return str(value)
    except Exception as exc:
        return f"Could not calculate that: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the live web for current information (news, companies, skills, salaries).

    Args:
        query: What to search for.
    """
    if not config.is_tavily_configured():
        return "Tavily is not configured. Add TAVILY_API_KEY to your .env file."
    try:
        raw = _tavily(max_results=3).invoke({"query": query})
        return _format_results(raw)
    except Exception as exc:
        return f"Web search failed: {exc}"


@tool
def job_search(role: str, skills: str, location: str = "remote") -> str:
    """Search the web for current job openings that match a role, skills, and location.

    Args:
        role: Job title to look for, e.g. "backend engineer".
        skills: Comma-separated skills from the CV, e.g. "Python, FastAPI, SQL".
        location: City, country, or "remote".
    """
    if not config.is_tavily_configured():
        return "Tavily is not configured. Add TAVILY_API_KEY to your .env file."

    query = f"{role} jobs {location} hiring {skills}"
    try:
        raw = _tavily(max_results=5).invoke({
            "query": query,
            "time_range": "month",
        })
        return _format_results(raw)
    except Exception as exc:
        return f"Job search failed: {exc}"


def get_tools() -> list:
    """Return the list of tools we hand to the agent."""
    return [calculator, web_search, job_search]
