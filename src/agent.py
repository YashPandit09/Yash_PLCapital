"""
agent.py – PortfolioAgent: routes questions to the appropriate tool.

Uses Gemini to classify questions as either sql_query or exposure_calculator,
then delegates to the corresponding tool. Produces a human-readable answer
from the tool output using deterministic formatting for numbers.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.db import get_connection  # ensure DB is initialised
from src.exposure_tool import ExposureResult, calculate_sector_exposure
from src.llm import generate, is_available
from src.sql_tool import SQLResult, run_sql_query

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class AgentResult:
    """Result from the PortfolioAgent."""
    question: str = ""
    tool_used: str = "none"         # "sql_query" | "exposure_calculator" | "none"
    tool_input: Any = None          # SQL string or params dict
    raw_result: Any = None          # SQLResult or ExposureResult
    answer: str = ""                # human-readable answer
    error: str = ""
    latency_ms: float = 0.0


# ──────────────────────────────────────────────
# Router prompt
# ──────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a portfolio analytics routing agent.
Given a user question, decide which tool to use.

TOOLS:
1. "sql_query" – For general database questions: counts, lists, aggregations, joins,
   portfolio values (quantity * price), performance metrics, risk metrics, transactions,
   holdings details, comparisons across portfolios, etc.
2. "exposure_calculator" – ONLY for questions specifically asking about sector exposure,
   sector breakdown, sector allocation, or sector weight distribution of a SPECIFIC portfolio.
   The question must reference a specific portfolio by name.

RULES:
- If the question asks about "sector exposure", "sector breakdown", "sector allocation",
  or "sector weights" for a specific portfolio → "exposure_calculator"
- Everything else → "sql_query"
- If the question is not about portfolio data at all → "none"

Output EXACTLY one JSON object on a single line:
{"tool": "<tool_name>", "portfolio_name": "<name or null>"}

If tool is "exposure_calculator", extract the portfolio name from the question.
If tool is "sql_query" or "none", set portfolio_name to null.

Examples:
Q: How many portfolios do we have?
{"tool": "sql_query", "portfolio_name": null}

Q: What are the sector exposures for the ESG Sustainable Fund?
{"tool": "exposure_calculator", "portfolio_name": "ESG Sustainable Fund"}

Q: Calculate the sector exposure breakdown for international equity
{"tool": "exposure_calculator", "portfolio_name": "international equity"}

Q: For each portfolio, show the total value of Technology sector holdings
{"tool": "sql_query", "portfolio_name": null}

Q: What is the weather today?
{"tool": "none", "portfolio_name": null}
"""


# ──────────────────────────────────────────────
# Answer formatters (deterministic – no LLM)
# ──────────────────────────────────────────────

def _format_sql_answer(result: SQLResult) -> str:
    """Format SQL result as a human-readable table."""
    if result.error:
        return f"Error: {result.error}"

    if not result.rows:
        return "Query returned no results."

    # Single value
    if len(result.rows) == 1 and len(result.columns) == 1:
        val = result.rows[0][0]
        if isinstance(val, float):
            return f"{val:,.2f}"
        return str(val)

    # Build fixed-width table
    lines: list[str] = []

    # Calculate column widths
    col_widths: list[int] = []
    for i, col in enumerate(result.columns):
        max_w = len(col)
        for row in result.rows:
            cell = _format_cell(row[i])
            max_w = max(max_w, len(cell))
        col_widths.append(min(max_w, 40))  # cap at 40 chars

    # Header
    header = " | ".join(
        col.ljust(col_widths[i]) for i, col in enumerate(result.columns)
    )
    lines.append(header)
    lines.append("-+-".join("-" * w for w in col_widths))

    # Data rows
    for row in result.rows:
        row_str = " | ".join(
            _format_cell(row[i]).ljust(col_widths[i])
            for i in range(len(result.columns))
        )
        lines.append(row_str)

    if result.truncated:
        lines.append(f"... (showing {result.row_count} of {result.row_count}+ rows)")

    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    """Format a single cell value for display."""
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _format_exposure_answer(result: ExposureResult) -> str:
    """Format exposure result as a human-readable table."""
    if result.error:
        return result.error

    lines: list[str] = [
        f"Sector Exposure for: {result.portfolio_name}",
        f"(Equity weight: {result.equity_weight_total}% of portfolio"
        f" | Bonds excluded: {result.bonds_excluded})",
        "",
    ]

    # Table header
    lines.append(f"{'Sector':<28} {'% of Equity':>12} {'% of Portfolio':>15}")
    lines.append("-" * 58)

    for sector in result.exposure_pct_of_equity:
        eq_pct = result.exposure_pct_of_equity[sector]
        port_pct = result.exposure_pct_of_portfolio.get(sector, 0.0)
        lines.append(f"{sector:<28} {eq_pct:>11.2f}% {port_pct:>14.2f}%")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Portfolio Agent
# ──────────────────────────────────────────────

class PortfolioAgent:
    """Routes portfolio questions to the appropriate tool and formats answers."""

    def __init__(self) -> None:
        # Ensure DB is initialised eagerly
        get_connection()
        # Simple in-memory cache
        self._cache: dict[str, AgentResult] = {}

    def answer_question(self, question: str) -> AgentResult:
        """Answer a portfolio analytics question.

        Args:
            question: Natural-language question.

        Returns:
            AgentResult with the tool used, raw result, and formatted answer.
        """
        start = time.perf_counter()

        # Input validation
        if not question or not question.strip():
            return AgentResult(
                question=question,
                error="Please provide a non-empty question.",
            )

        q = question.strip()

        # Check cache
        cache_key = hashlib.md5(q.lower().encode()).hexdigest()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            elapsed = (time.perf_counter() - start) * 1000
            cached.latency_ms = elapsed
            logger.info("Cache hit for question (%.1fms)", elapsed)
            return cached

        # Route the question
        tool_name, portfolio_name = self._route(q)
        logger.info("Routed to tool=%s, portfolio=%s", tool_name, portfolio_name)

        result: AgentResult

        if tool_name == "exposure_calculator":
            result = self._run_exposure(q, portfolio_name)
        elif tool_name == "sql_query":
            result = self._run_sql(q)
        else:
            result = AgentResult(
                question=q,
                tool_used="none",
                answer="I can only answer questions about portfolio data. "
                       "Please ask about portfolios, holdings, securities, "
                       "performance, risk metrics, or sector exposures.",
            )

        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed
        logger.info("Answered in %.1fms using %s", elapsed, result.tool_used)

        # Cache the result
        self._cache[cache_key] = result
        return result

    def _route(self, question: str) -> tuple[str, Optional[str]]:
        """Route question based on domain routing rule, preserving API quota."""
        # 1. Check exposure keywords per assignment rules
        tool, pname = self._heuristic_route(question)
        if tool == "exposure_calculator":
            return tool, pname

        # 2. Check general financial/portfolio data queries
        q_lower = question.lower()
        sql_signals = [
            "how many", "what", "which", "show", "list", "total", "average", "avg",
            "top", "count", "find", "for each", "portfolio", "holding", "security",
            "aum", "price", "sector", "risk", "performance", "return", "transaction",
            "cost", "diversification", "fund"
        ]
        if any(sig in q_lower for sig in sql_signals):
            return "sql_query", None

        # 3. For ambiguous queries, use Gemini router if available
        if is_available():
            try:
                import json
                raw = generate(question, system_instruction=_ROUTER_SYSTEM)
                text = raw.strip()
                if text.startswith("```"):
                    lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                data = json.loads(text)
                t = data.get("tool", "sql_query")
                return (t, data.get("portfolio_name")) if t in ("sql_query", "exposure_calculator", "none") else ("sql_query", None)
            except Exception as exc:
                logger.warning("Router LLM failed, using fallback: %s", exc)

        return "none", None

    @staticmethod
    def _heuristic_route(question: str) -> tuple[str, Optional[str]]:
        """Simple keyword-based routing fallback."""
        q_lower = question.lower()
        exposure_keywords = [
            "sector exposure", "sector breakdown", "sector allocation",
            "sector weight", "exposure breakdown",
        ]
        if any(kw in q_lower for kw in exposure_keywords):
            # Try to extract portfolio name (rough heuristic)
            # Look for text after "for" or "of"
            import re
            match = re.search(r"(?:for|of)\s+(?:the\s+)?(.+?)(?:\?|$)", q_lower)
            pname = match.group(1).strip().rstrip("?. ") if match else None
            return "exposure_calculator", pname
        return "sql_query", None

    def _run_sql(self, question: str) -> AgentResult:
        """Run the SQL tool and format the answer."""
        sql_result = run_sql_query(question)
        answer = _format_sql_answer(sql_result)

        return AgentResult(
            question=question,
            tool_used="sql_query",
            tool_input=sql_result.sql,
            raw_result=sql_result,
            answer=answer,
            error=sql_result.error,
        )

    def _run_exposure(self, question: str, portfolio_name: Optional[str]) -> AgentResult:
        """Run the exposure calculator tool and format the answer."""
        if not portfolio_name:
            return AgentResult(
                question=question,
                tool_used="exposure_calculator",
                error="Could not determine which portfolio to analyze. "
                      "Please specify a portfolio name.",
            )

        exp_result = calculate_sector_exposure(portfolio_name=portfolio_name)
        answer = _format_exposure_answer(exp_result)

        return AgentResult(
            question=question,
            tool_used="exposure_calculator",
            tool_input={"portfolio_name": portfolio_name, "resolved_name": exp_result.portfolio_name},
            raw_result=exp_result,
            answer=answer,
            error=exp_result.error,
        )

