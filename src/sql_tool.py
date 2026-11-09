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

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "What is the total dollar volume for BUY transactions?",
        "SELECT SUM(quantity * price) AS total_buy_volume FROM transactions WHERE transaction_type = 'BUY'",
    ),
    (
        "List all portfolios and their latest beta from the risk metrics table",
        "SELECT p.portfolio_name, r.beta, r.calculation_date FROM portfolios p JOIN risk_metrics r ON p.portfolio_id = r.portfolio_id WHERE r.calculation_date = (SELECT MAX(rm.calculation_date) FROM risk_metrics rm WHERE rm.portfolio_id = r.portfolio_id)",
    ),
    (
        "Find the top 5 largest market cap stocks",
        "SELECT symbol, company_name, market_cap FROM securities WHERE asset_type = 'Stock' ORDER BY market_cap DESC LIMIT 5",
    ),
    (
        "Show portfolios with a Sharpe ratio greater than 1.0 along with the performance date",
        "SELECT p.portfolio_name, pp.sharpe_ratio, pp.performance_date FROM portfolios p JOIN portfolio_performance pp ON p.portfolio_id = pp.portfolio_id WHERE pp.sharpe_ratio > 1.0",
    ),
    (
        "What is the average coupon rate for bonds maturing after 2025?",
        "SELECT AVG(coupon_rate) AS avg_coupon FROM securities WHERE asset_type = 'Bond' AND maturity_date > '2025-12-31'",
    ),
]

_SYSTEM_PROMPT_TEMPLATE = """You are a SQLite SQL expert. Given a natural-language question about portfolio data,
generate ONLY the SQL query — no explanation, no markdown fences, no comments.

RULES:
1. Output ONLY a single read-only SELECT statement (or WITH ... SELECT). Use SQLite syntax.
2. Return ONLY the raw SQL. Do not include markdown code blocks or explanations.
3. Return only the columns the question specifically asks for; do not include surrogate id columns (such as primary or foreign key IDs) unless explicitly requested.
4. Holding and portfolio market value is calculated as SUM(quantity * current_price) over holdings; use portfolios.total_aum only when the question specifically asks for AUM.
5. For "percentage of total" or "share" questions, return the component value, the total value, and the calculated percentage.
6. Equities have asset_type = 'Stock'. Bonds have asset_type = 'Bond' and sector_id is NULL.
7. Always qualify column names with table aliases when joining tables to avoid ambiguity.
8. Use ORDER BY and LIMIT when ranking or when the question asks for top/bottom or highest/lowest values.
9. Use standard SQLite aggregation functions (e.g., SUM, AVG, COUNT, MIN, MAX) and GROUP BY appropriately.

{schema}

FEW-SHOT EXAMPLES:

{few_shots}
"""


def _build_system_prompt() -> str:
    """Build the system prompt with the runtime schema description and few-shot examples."""
    schema = get_schema_description()
    few_shots_text = "\n\n".join(
        f"Q: {q}\nA: {sql}" for q, sql in FEW_SHOT_EXAMPLES
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(schema=schema, few_shots=few_shots_text)


def _strip_sql(raw: str) -> str:
    """Extract and clean SQL from LLM output, stripping fences and conversational prose."""
    text = raw.strip()
    # Check for markdown code fence anywhere in the output
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        # If model outputs prose followed by SQL, find start of SELECT or WITH
        select_match = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, re.IGNORECASE)
        if select_match:
            text = select_match.group(0).strip()

    # Remove trailing semicolon
    return text.rstrip(";").strip()


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

