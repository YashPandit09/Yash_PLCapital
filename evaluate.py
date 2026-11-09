"""
evaluate.py – Agent evaluator for Portfolio Analytics Agent.

Runs ground-truth questions through the agent and validates results.

Usage:
    python evaluate.py                           # full run
    python evaluate.py --offline                  # no API key needed
    python evaluate.py --verbose                  # detailed output
    python evaluate.py --ids 1,2,3                # specific questions only
    python evaluate.py --file other.json          # custom ground-truth file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import AgentResult, PortfolioAgent
from src.db import get_connection
from src.exposure_tool import ExposureResult
from src.sql_tool import SQLResult, execute_ground_truth_sql


# ──────────────────────────────────────────────
# Query Plan & Performance Cost Analysis
# ──────────────────────────────────────────────

def analyze_query_plan(sql: str) -> dict[str, Any]:
    """Analyze query execution plan via EXPLAIN QUERY PLAN.

    Flags full table scans vs indexed lookups and temp B-trees.
    """
    conn = get_connection()
    try:
        clean_sql = sql.strip().rstrip(";")
        plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {clean_sql}").fetchall()
        details = [r[3] for r in plan_rows]
        full_scans = [
            d for d in details
            if "SCAN" in d and "USING INDEX" not in d and "USING COVERING INDEX" not in d
        ]
        index_uses = [
            d for d in details
            if "USING INDEX" in d or "USING COVERING INDEX" in d
        ]
        temp_btrees = [d for d in details if "TEMP B-TREE" in d]
        return {
            "plan_steps": details,
            "has_full_table_scan": len(full_scans) > 0,
            "full_scans": full_scans,
            "index_lookups": len(index_uses),
            "temp_btrees": len(temp_btrees),
        }
    except Exception as exc:
        return {"error": str(exc), "plan_steps": [], "has_full_table_scan": False}


# ──────────────────────────────────────────────
# Result comparison helpers
# ──────────────────────────────────────────────

def _normalize_value(val: Any) -> Any:
    """Normalize a single value for comparison."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return int(f) if f.is_integer() else round(f, 2)
    if isinstance(val, str):
        try:
            f = float(val)
            return int(f) if f.is_integer() else round(f, 2)
        except (ValueError, TypeError):
            return val.strip()
    return val


def _normalize_row(row: tuple) -> tuple:
    """Normalize all values in a row."""
    return tuple(_normalize_value(v) for v in row)


def _compare_results(
    candidate: SQLResult,
    ground_truth: SQLResult,
    gt_sql: str,
) -> tuple[bool, str]:
    """Compare candidate SQL results against ground-truth results (Strict Mode).

    Rules:
    - Normalize numbers to 2 decimal places.
    - Ignore column names/aliases.
    - Compare as multisets of row-tuples.
    - If ground-truth SQL has ORDER BY, also check row order.
    - Exact column count match required.
    """
    if candidate.error:
        return False, f"Candidate query error: {candidate.error}"

    if ground_truth.error:
        return False, f"Ground truth query error: {ground_truth.error}"

    # Column count check
    if len(candidate.columns) != len(ground_truth.columns):
        return False, (
            f"Column count mismatch: got {len(candidate.columns)} "
            f"({candidate.columns}), expected {len(ground_truth.columns)} "
            f"({ground_truth.columns})"
        )

    # Normalize rows
    cand_rows = [_normalize_row(r) for r in candidate.rows]
    gt_rows = [_normalize_row(r) for r in ground_truth.rows]

    # Check if ORDER BY is present
    has_order = "ORDER BY" in gt_sql.upper()

    if has_order:
        # Ordered comparison
        if cand_rows != gt_rows:
            return False, (
                f"Row mismatch (ordered). "
                f"Got {len(cand_rows)} rows, expected {len(gt_rows)}.\n"
                f"  Candidate columns: {candidate.columns}\n"
                f"  First candidate rows: {cand_rows[:5]}\n"
                f"  Expected columns:  {ground_truth.columns}\n"
                f"  First expected rows:  {gt_rows[:5]}"
            )
    else:
        # Multiset comparison
        cand_sorted = sorted(cand_rows, key=str)
        gt_sorted = sorted(gt_rows, key=str)
        if cand_sorted != gt_sorted:
            return False, (
                f"Row mismatch (unordered). "
                f"Got {len(cand_rows)} rows, expected {len(gt_rows)}.\n"
                f"  Candidate columns: {candidate.columns}\n"
                f"  Candidate (sorted): {cand_sorted[:5]}\n"
                f"  Expected columns:  {ground_truth.columns}\n"
                f"  Expected  (sorted): {gt_sorted[:5]}"
            )

    return True, ""


def _compare_lenient(
    candidate: SQLResult,
    ground_truth: SQLResult,
    gt_sql: str,
    question: str,
) -> tuple[bool, str]:
    """Compare candidate SQL results against ground-truth under documented Lenient Rules:
    (a) Extra columns are allowed if every expected column's values appear in some result column.
    (b) Row order is compared only if ground-truth SQL has LIMIT or the question contains
        top/highest/lowest/rank/sort/order.
    """
    if candidate.error:
        return False, f"Candidate query error: {candidate.error}"

    if ground_truth.error:
        return False, f"Ground truth query error: {ground_truth.error}"

    cand_rows = [_normalize_row(r) for r in candidate.rows]
    gt_rows = [_normalize_row(r) for r in ground_truth.rows]

    if not cand_rows and not gt_rows:
        return True, ""
    if not cand_rows or not gt_rows:
        return False, f"Row count mismatch: got {len(cand_rows)} rows, expected {len(gt_rows)}"

    m = len(candidate.columns)
    k = len(ground_truth.columns)

    if m < k:
        return False, (
            f"Candidate has fewer columns ({m}) than expected ({k}). "
            f"Candidate columns: {candidate.columns}, Expected: {ground_truth.columns}"
        )

    if len(cand_rows) != len(gt_rows):
        return False, f"Row count mismatch: got {len(cand_rows)} rows, expected {len(gt_rows)}"

    # Rule (b): Row order is compared only if ground-truth SQL has LIMIT
    # or the question contains top/highest/lowest/rank/sort/order
    order_keywords = ["top", "highest", "lowest", "rank", "sort", "order"]
    q_lower = question.lower()
    has_limit = "LIMIT" in gt_sql.upper()
    must_order = has_limit or any(kw in q_lower for kw in order_keywords)

    import itertools
    gt_sorted = sorted(gt_rows, key=str) if not must_order else None

    # Rule (a): Test all length-k permutations of candidate column indices
    for col_indices in itertools.permutations(range(m), k):
        proj_rows = [tuple(r[idx] for idx in col_indices) for r in cand_rows]
        if must_order:
            if proj_rows == gt_rows:
                return True, ""
        else:
            if sorted(proj_rows, key=str) == gt_sorted:
                return True, ""

    return False, (
        f"Lenient match failed: no candidate column projection matched expected values.\n"
        f"  Candidate columns ({m}): {candidate.columns}\n"
        f"  Expected columns ({k}): {ground_truth.columns}"
    )


def _compute_independent_exposure(
    gt_portfolio_name: str,
) -> tuple[Optional[dict[str, float]], Optional[str]]:
    """Compute expected equity sector exposure independently via direct SQL query on DB.

    Does NOT use src.exposure_tool. Directly queries holdings, securities, sectors, portfolios.
    Excludes bonds (asset_type != 'Stock' or sector_id IS NULL), renormalizes equity weights to 100%.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT portfolio_id, portfolio_name FROM portfolios WHERE LOWER(portfolio_name) = LOWER(?)",
        (gt_portfolio_name.strip(),),
    ).fetchone()
    if not row:
        return None, f"Portfolio '{gt_portfolio_name}' not found in database"

    pid, official_name = row[0], row[1]

    # Query equity holdings
    query = """
        SELECT sec.sector_name, SUM(h.current_weight) as sector_weight
        FROM holdings h
        JOIN securities s ON h.security_id = s.security_id
        JOIN sectors sec ON s.sector_id = sec.sector_id
        WHERE h.portfolio_id = ?
          AND s.asset_type = 'Stock'
          AND s.sector_id IS NOT NULL
        GROUP BY sec.sector_name
    """
    rows = conn.execute(query, (pid,)).fetchall()
    if not rows:
        return None, f"Portfolio '{official_name}' has no equity holdings"

    total_weight = sum(r[1] for r in rows)
    if total_weight <= 0:
        return None, "Total equity weight is zero"

    # Renormalize to 100% and round to 2 decimals
    result = {}
    for sector_name, weight in rows:
        pct = round((weight / total_weight) * 100.0, 2)
        result[sector_name] = pct

    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return sorted_result, None


def _verify_exposure(
    agent_result: AgentResult,
    gt_portfolio_name: str,
) -> tuple[bool, str]:
    """Verify exposure calculator results independently without calling src.exposure_tool."""
    raw = agent_result.raw_result
    if not isinstance(raw, ExposureResult):
        # If agent errored, check if it's a valid error (e.g. bond-only portfolio)
        if agent_result.error:
            return False, f"Agent error: {agent_result.error}"
        return False, "Agent did not return ExposureResult"

    if raw.error:
        return False, f"Exposure calculation error: {raw.error}"

    # Recompute independently using direct SQL query
    independent, err = _compute_independent_exposure(gt_portfolio_name)
    if err:
        return False, f"Independent calculation error: {err}"

    # Check portfolio resolved correctly
    if raw.portfolio_name.lower() != gt_portfolio_name.lower():
        return False, (
            f"Portfolio name mismatch: agent='{raw.portfolio_name}', "
            f"expected='{gt_portfolio_name}'"
        )

    # Compare exposure values within tolerance
    for sector, expected_val in independent.items():
        agent_val = raw.exposure_pct_of_equity.get(sector, -999)
        if abs(agent_val - expected_val) > 0.05:
            return False, (
                f"Exposure mismatch for {sector}: "
                f"agent={agent_val}, expected={expected_val}"
            )

    # Check no extra sectors
    extra = set(raw.exposure_pct_of_equity) - set(independent.keys())
    if extra:
        return False, f"Agent returned extra sectors: {extra}"

    return True, ""


# ──────────────────────────────────────────────
# Offline SQL runner (replaces LLM)
# ──────────────────────────────────────────────

class OfflineAgent:
    """Fake agent that uses ground-truth SQL instead of the LLM."""

    def __init__(self) -> None:
        get_connection()
        self._real_agent = PortfolioAgent()

    def answer_question_with_gt(
        self, question: dict[str, Any]
    ) -> AgentResult:
        """Answer using ground-truth data directly."""
        start = time.perf_counter()
        q_text = question["question"]
        q_type = question["type"]
        gt = question["ground_truth"]

        if q_type == "text2sql":
            sql = gt["sql_query"]
            result = execute_ground_truth_sql(sql)
            elapsed = (time.perf_counter() - start) * 1000
            return AgentResult(
                question=q_text,
                tool_used="sql_query",
                tool_input=sql,
                raw_result=result,
                answer="(offline)",
                error=result.error,
                latency_ms=elapsed,
            )
        elif q_type == "exposure_calculator":
            return self._real_agent.answer_question(q_text)
        else:
            elapsed = (time.perf_counter() - start) * 1000
            return AgentResult(
                question=q_text,
                tool_used="none",
                answer="Unknown question type",
                latency_ms=elapsed,
            )


# ──────────────────────────────────────────────
# Import here to avoid circular issues at module level
# ──────────────────────────────────────────────
from src.exposure_tool import ExposureResult


# ──────────────────────────────────────────────
# Main evaluator
# ──────────────────────────────────────────────

def evaluate(
    gt_file: Path,
    offline: bool = False,
    verbose: bool = False,
    question_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Run evaluation and return a report dict."""
    with open(gt_file, encoding="utf-8") as f:
        gt_data = json.load(f)

    questions = gt_data["questions"]
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]

    agent: Any
    if offline:
        agent = OfflineAgent()
    else:
        agent = PortfolioAgent()

    results: list[dict[str, Any]] = []
    strict_pass_count = 0
    strict_fail_count = 0
    lenient_pass_count = 0
    lenient_fail_count = 0

    # Print header
    print()
    print("=" * 80)
    print(f"  Portfolio Analytics Agent – Evaluation ({'OFFLINE' if offline else 'LIVE'})")
    print(f"  Questions: {len(questions)}")
    print("=" * 80)
    print()

    for q in questions:
        qid = q["id"]
        q_type = q["type"]
        difficulty = q.get("difficulty", "unknown")
        q_text = q["question"]
        gt = q["ground_truth"]

        print(f"Q{qid} [{difficulty}] ({q_type}): {q_text}")

        # Run agent
        if offline:
            agent_result = agent.answer_question_with_gt(q)
        else:
            agent_result = agent.answer_question(q_text)

        # Validate tool routing
        expected_tool = "sql_query" if q_type == "text2sql" else "exposure_calculator"
        tool_ok = agent_result.tool_used == expected_tool

        strict_passed = False
        strict_reason = ""
        lenient_passed = False
        lenient_reason = ""

        if not tool_ok:
            strict_reason = (
                f"Wrong tool: used '{agent_result.tool_used}', "
                f"expected '{expected_tool}'"
            )
            lenient_reason = strict_reason
        elif q_type == "text2sql":
            # Execute ground-truth SQL
            gt_result = execute_ground_truth_sql(gt["sql_query"])
            candidate = agent_result.raw_result
            if isinstance(candidate, SQLResult):
                strict_passed, strict_reason = _compare_results(candidate, gt_result, gt["sql_query"])
                lenient_passed, lenient_reason = _compare_lenient(candidate, gt_result, gt["sql_query"], q_text)
            else:
                strict_reason = "Agent did not return SQLResult"
                lenient_reason = strict_reason
        elif q_type == "exposure_calculator":
            gt_pname = gt["parameters"]["portfolio_name"]
            tool_input = agent_result.tool_input or {}
            resolved = tool_input.get("resolved_name", "") if isinstance(tool_input, dict) else ""
            if resolved and resolved != gt_pname:
                pass
            strict_passed, strict_reason = _verify_exposure(agent_result, gt_pname)
            lenient_passed = strict_passed
            lenient_reason = strict_reason

        status = "PASS" if strict_passed else "FAIL"
        if strict_passed:
            strict_pass_count += 1
        else:
            strict_fail_count += 1

        if lenient_passed:
            lenient_pass_count += 1
        else:
            lenient_fail_count += 1

        status_icon = "✓" if strict_passed else "✗"
        if strict_passed:
            print(f"  {status_icon} PASS | tool={agent_result.tool_used} | {agent_result.latency_ms:.0f}ms")
        elif lenient_passed:
            print(f"  {status_icon} FAIL [Lenient: PASS] | tool={agent_result.tool_used} | {agent_result.latency_ms:.0f}ms")
        else:
            print(f"  {status_icon} FAIL | tool={agent_result.tool_used} | {agent_result.latency_ms:.0f}ms")

        # Query plan cost proxy (bonus enhancement)
        cost_info: Optional[dict[str, Any]] = None
        if q_type == "text2sql":
            target_sql = agent_result.tool_input or gt.get("sql_query", "")
            if target_sql:
                cost_info = analyze_query_plan(str(target_sql))
                if cost_info.get("has_full_table_scan"):
                    scans_str = ", ".join(cost_info.get("full_scans", []))
                    print(f"    [Perf Alert] Full table scan detected: {scans_str}")
                else:
                    idx_cnt = cost_info.get("index_lookups", 0)
                    print(f"    [Perf Optimal] Index lookups: {idx_cnt} index scan(s)")

        if not strict_passed and strict_reason:
            print(f"    Reason: {strict_reason}")
            if lenient_passed:
                print(f"    [Lenient Evaluation: PASS – expected columns/unordered values matched]")

        if verbose:
            if agent_result.tool_used == "sql_query" and agent_result.tool_input:
                print(f"    Agent SQL: {agent_result.tool_input}")
                print(f"    GT SQL:    {gt.get('sql_query', 'N/A')}")
            if cost_info and cost_info.get("plan_steps"):
                print(f"    Plan Steps: {cost_info['plan_steps']}")
            if agent_result.error:
                print(f"    Error: {agent_result.error}")

        print()

        results.append({
            "id": qid,
            "question": q_text,
            "type": q_type,
            "difficulty": difficulty,
            "status": status,
            "strict_passed": strict_passed,
            "lenient_passed": lenient_passed,
            "tool_used": agent_result.tool_used,
            "expected_tool": expected_tool,
            "latency_ms": round(agent_result.latency_ms, 1),
            "reason": strict_reason if not strict_passed else "",
            "lenient_reason": lenient_reason if not lenient_passed else "",
            "agent_sql": agent_result.tool_input if q_type == "text2sql" else None,
            "cost_proxy": cost_info,
        })

        # Polite delay between live queries to respect Gemini Free Tier 5 RPM quota
        if not offline and agent_result.tool_used == "sql_query" and q != questions[-1]:
            time.sleep(10.0)

    # ── Summary ──
    total = strict_pass_count + strict_fail_count
    strict_acc = (strict_pass_count / total * 100) if total > 0 else 0
    lenient_acc = (lenient_pass_count / total * 100) if total > 0 else 0

    # By type
    sql_results = [r for r in results if r["type"] == "text2sql"]
    exp_results = [r for r in results if r["type"] == "exposure_calculator"]
    sql_strict_pass = sum(1 for r in sql_results if r["strict_passed"])
    sql_lenient_pass = sum(1 for r in sql_results if r["lenient_passed"])
    exp_pass = sum(1 for r in exp_results if r["strict_passed"])

    # By difficulty
    diff_stats: dict[str, tuple[int, int]] = {}
    for r in results:
        d = r["difficulty"]
        p, t = diff_stats.get(d, (0, 0))
        diff_stats[d] = (p + (1 if r["strict_passed"] else 0), t + 1)

    # Query performance metrics
    full_scan_count = sum(
        1 for r in sql_results
        if r.get("cost_proxy") and r["cost_proxy"].get("has_full_table_scan")
    )
    indexed_count = len(sql_results) - full_scan_count

    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Strict Accuracy (Headline):  {strict_pass_count}/{total} ({strict_acc:.0f}%)")
    print(f"  Lenient Accuracy (Flexible): {lenient_pass_count}/{total} ({lenient_acc:.0f}%)")
    print(f"  text2sql (Strict):           {sql_strict_pass}/{len(sql_results)}")
    print(f"  text2sql (Lenient):          {sql_lenient_pass}/{len(sql_results)}")
    print(f"  exposure_calculator:         {exp_pass}/{len(exp_results)}")
    for d in sorted(diff_stats):
        p, t = diff_stats[d]
        print(f"  {d} (Strict):        {p}/{t}")
    if sql_results:
        print(f"  SQL Optimization:            {indexed_count}/{len(sql_results)} queries optimal, {full_scan_count} flagged for table scans")
    print("=" * 80)
    print()

    report = {
        "summary": {
            "total": total,
            "passed": strict_pass_count,
            "failed": strict_fail_count,
            "strict_passed": strict_pass_count,
            "strict_failed": strict_fail_count,
            "strict_accuracy_pct": round(strict_acc, 1),
            "lenient_passed": lenient_pass_count,
            "lenient_failed": lenient_fail_count,
            "lenient_accuracy_pct": round(lenient_acc, 1),
            "text2sql_strict": f"{sql_strict_pass}/{len(sql_results)}",
            "text2sql_lenient": f"{sql_lenient_pass}/{len(sql_results)}",
            "exposure_calculator": f"{exp_pass}/{len(exp_results)}",
            "by_difficulty_strict": {d: f"{p}/{t}" for d, (p, t) in diff_stats.items()},
            "sql_performance": {
                "total_sql_queries": len(sql_results),
                "indexed_queries": indexed_count,
                "full_table_scans": full_scan_count,
            },
        },
        "results": results,
    }

    # Save report
    report_path = PROJECT_ROOT / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Portfolio Analytics Agent"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="ground_truth_dataset.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with ground-truth SQL (no API key needed)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output per question",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated question IDs to run (e.g. 1,2,3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    gt_path = Path(args.file)
    if not gt_path.is_absolute():
        gt_path = PROJECT_ROOT / gt_path

    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)

    question_ids = None
    if args.ids:
        question_ids = [int(x.strip()) for x in args.ids.split(",")]

    report = evaluate(
        gt_file=gt_path,
        offline=args.offline,
        verbose=args.verbose,
        question_ids=question_ids,
    )

    # Exit code 1 if any headline strict failure
    if report["summary"].get("strict_failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

