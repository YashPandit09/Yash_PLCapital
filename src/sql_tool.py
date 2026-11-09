"""
sql_tool.py – Text-to-SQL tool for Portfolio Analytics Agent.

Converts natural-language questions into SQL queries using Gemini,
validates them for safety, executes them against the in-memory SQLite DB,
and returns structured results.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from src.db import get_connection, get_schema_description
from src.llm import generate

logger = logging.getLogger(__name__)

_ROW_LIMIT = 1000

# ──────────────────────────────────────────────
# SQL Safety Validation
# ──────────────────────────────────────────────

_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|REPLACE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL for safety.

    Returns:
        (is_valid, error_message). error_message is empty if valid.
    """
    stripped = sql.strip().rstrip(";").strip()

    if not stripped:
        return False, "Empty SQL statement."

    # Must start with SELECT or WITH
    upper = stripped.upper().lstrip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "Only SELECT / WITH queries are allowed."

    # Block dangerous keywords
    match = _BLOCKED_KEYWORDS.search(stripped)
    if match:
        return False, f"Blocked keyword detected: {match.group()}"

    # Single statement only: no semicolon followed by more SQL
    # Split on semicolons and check for non-empty parts after the first
    parts = [p.strip() for p in stripped.split(";") if p.strip()]
    if len(parts) > 1:
        return False, "Only a single SQL statement is allowed."

    return True, ""


def _explain_check(conn: sqlite3.Connection, sql: str) -> tuple[bool, str]:
    """Run EXPLAIN to validate SQL syntax before execution."""
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True, ""
    except sqlite3.Error as exc:
        return False, str(exc)


# ──────────────────────────────────────────────
# System prompt for Text-to-SQL
# ──────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """You are a SQLite SQL expert. Given a natural-language question about portfolio data,
generate ONLY the SQL query — no explanation, no markdown fences, no comments.

RULES:
1. Output ONLY a single SELECT statement (or WITH ... SELECT). SQLite syntax.
2. No trailing semicolons or explanations after the SQL.
3. "Portfolio value" or "holding value" = SUM(holdings.quantity * securities.current_price)
4. Use exact portfolio_name / sector_name text from the DB. Use LIKE / LOWER() for loose matching.
5. Bonds (asset_type='Bond') have NULL sector_id. Use INNER JOIN to sectors ONLY when sector info is needed.
6. For "total AUM", use portfolios.total_aum column directly.
7. Always qualify ambiguous column names with table aliases.
8. Use ORDER BY when the question asks for "top", "highest", "lowest", or ranking.
9. Use LIMIT when the question specifies a number of results.

{schema}

FEW-SHOT EXAMPLES:

Q: How many portfolios do we have in total?
A: SELECT COUNT(*) FROM portfolios

Q: What are the names of all active portfolios?
A: SELECT portfolio_name FROM portfolios WHERE status = 'Active'

Q: Which securities are in the Technology sector?
A: SELECT s.symbol, s.company_name FROM securities s JOIN sectors sec ON s.sector_id = sec.sector_id WHERE sec.sector_name = 'Technology'

Q: Show me the top 5 holdings by cost basis in the Growth Equity Fund
A: SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis FROM holdings h JOIN securities s ON h.security_id = s.security_id JOIN portfolios p ON h.portfolio_id = p.portfolio_id WHERE p.portfolio_name = 'Growth Equity Fund' ORDER BY h.cost_basis DESC LIMIT 5
"""


def _build_system_prompt() -> str:
    """Build the system prompt with the current schema description."""
    schema = get_schema_description()
    return _SYSTEM_PROMPT_TEMPLATE.format(schema=schema)


def _strip_sql(raw: str) -> str:
    """Strip markdown fences, trailing semicolons, and whitespace from LLM output."""
    text = raw.strip()
    # Remove ```sql ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Remove trailing semicolon
    text = text.rstrip(";").strip()
    return text


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────

@dataclass
class SQLResult:
    """Result of a SQL query tool invocation."""
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    error: str = ""
    truncated: bool = False


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run_sql_query(question: str) -> SQLResult:
    """Convert a natural-language question to SQL, validate, execute, and return results.

    Includes one self-repair retry if the first generated SQL fails.
    """
    conn = get_connection()
    system_prompt = _build_system_prompt()

    # ── Generate SQL via LLM ──
    try:
        raw_sql = generate(question, system_instruction=system_prompt)
    except RuntimeError as exc:
        return SQLResult(error=f"LLM error: {exc}")

    sql = _strip_sql(raw_sql)
    logger.info("Generated SQL: %s", sql)

    # ── Safety validation ──
    is_valid, err_msg = validate_sql(sql)
    if not is_valid:
        return SQLResult(sql=sql, error=f"SQL validation failed: {err_msg}")

    # ── EXPLAIN check ──
    explain_ok, explain_err = _explain_check(conn, sql)
    if not explain_ok:
        logger.warning("EXPLAIN failed, attempting self-repair: %s", explain_err)
        sql = _self_repair(question, sql, explain_err, system_prompt)
        if sql is None:
            return SQLResult(error=f"SQL syntax error (repair failed): {explain_err}")

    # ── Execute (with read-only safeguard) ──
    return _execute_sql(conn, sql)


def execute_ground_truth_sql(sql: str) -> SQLResult:
    """Execute a known-good SQL string directly (for evaluation / offline mode)."""
    conn = get_connection()
    stripped = sql.strip().rstrip(";").strip()
    return _execute_sql(conn, stripped)


def _execute_sql(conn: sqlite3.Connection, sql: str) -> SQLResult:
    """Execute SQL on a read-only cursor and return results."""
    try:
        # Enforce read-only at connection level
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(_ROW_LIMIT + 1)
        truncated = len(rows) > _ROW_LIMIT
        if truncated:
            rows = rows[:_ROW_LIMIT]
        return SQLResult(
            sql=sql,
            columns=columns,
            rows=[tuple(r) for r in rows],
            row_count=len(rows),
            truncated=truncated,
        )
    except sqlite3.Error as exc:
        return SQLResult(sql=sql, error=f"SQL execution error: {exc}")
    finally:
        try:
            conn.execute("PRAGMA query_only = OFF")
        except sqlite3.Error:
            pass


def _self_repair(
    question: str, failed_sql: str, error: str, system_prompt: str
) -> Optional[str]:
    """Attempt one LLM self-repair cycle for a failed SQL query."""
    repair_prompt = (
        f"The following SQL query failed with an error.\n\n"
        f"Question: {question}\n"
        f"Failed SQL: {failed_sql}\n"
        f"Error: {error}\n\n"
        f"Generate a corrected SQL query. Output ONLY the SQL, nothing else."
    )
    try:
        raw = generate(repair_prompt, system_instruction=system_prompt)
        repaired = _strip_sql(raw)
        logger.info("Repaired SQL: %s", repaired)

        is_valid, err_msg = validate_sql(repaired)
        if not is_valid:
            logger.warning("Repaired SQL also invalid: %s", err_msg)
            return None

        conn = get_connection()
        ok, _ = _explain_check(conn, repaired)
        return repaired if ok else None

    except RuntimeError:
        return None

