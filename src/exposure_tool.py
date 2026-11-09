"""
exposure_tool.py – Sector exposure calculator for Portfolio Analytics Agent.

Pure Python/pandas calculation. NO LLM calls inside.
Calculates sector exposure for a given portfolio, excluding bonds (asset_type='Bond').
"""

from __future__ import annotations

import difflib
import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union

from src.db import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ExposureResult(Mapping):
    """Result of a sector exposure calculation.

    Implements Mapping (dict-like access) for seamless compatibility
    with both dictionary indexing (``result['exposure_pct_of_equity']``)
    and attribute access (``result.exposure_pct_of_equity``).
    """
    portfolio_name: str = ""
    portfolio_id: int = 0
    exposure_pct_of_equity: dict[str, float] = field(default_factory=dict)
    exposure_pct_of_portfolio: dict[str, float] = field(default_factory=dict)
    equity_weight_total: float = 0.0
    bonds_excluded: int = 0
    error: str = ""

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from [
            "portfolio_name",
            "portfolio_id",
            "exposure_pct_of_equity",
            "exposure_pct_of_portfolio",
            "equity_weight_total",
            "bonds_excluded",
            "error",
        ]

    def __len__(self) -> int:
        return 7

    def to_dict(self) -> dict[str, Any]:
        """Return results as a plain dictionary."""
        return {k: getattr(self, k) for k in self}


def _get_portfolio_names(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Return all (portfolio_id, portfolio_name) pairs."""
    rows = conn.execute(
        "SELECT portfolio_id, portfolio_name FROM portfolios ORDER BY portfolio_id"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _resolve_portfolio(
    conn: sqlite3.Connection,
    name: Optional[str] = None,
    pid: Optional[int] = None,
) -> tuple[Optional[int], Optional[str], str]:
    """Resolve a portfolio by ID or name (fuzzy matching for names).

    Returns:
        (portfolio_id, portfolio_name, error_message)
    """
    all_portfolios = _get_portfolio_names(conn)
    all_names = [p[1] for p in all_portfolios]

    if pid is not None:
        for p_id, p_name in all_portfolios:
            if p_id == pid:
                return p_id, p_name, ""
        return None, None, (
            f"Portfolio with id={pid} not found. "
            f"Available portfolios: {', '.join(all_names)}"
        )

    if name is None or not name.strip():
        return None, None, (
            f"No portfolio specified. Available: {', '.join(all_names)}"
        )

    query = name.strip()

    # 1. Exact case-insensitive match
    for p_id, p_name in all_portfolios:
        if p_name.lower() == query.lower():
            return p_id, p_name, ""

    # 2. Substring token match: check if all words in query appear in a portfolio name
    query_tokens = query.lower().split()
    for p_id, p_name in all_portfolios:
        p_lower = p_name.lower()
        if all(tok in p_lower for tok in query_tokens):
            return p_id, p_name, ""

    # 3. Fuzzy match via difflib
    close = difflib.get_close_matches(
        query.lower(),
        [n.lower() for n in all_names],
        n=1,
        cutoff=0.5,
    )
    if close:
        for p_id, p_name in all_portfolios:
            if p_name.lower() == close[0]:
                return p_id, p_name, ""

    return None, None, (
        f"Could not resolve portfolio '{query}'. "
        f"Available portfolios: {', '.join(all_names)}"
    )


def calculate_sector_exposure(
    portfolio_name: Optional[str] = None,
    portfolio_id: Optional[int] = None,
) -> ExposureResult:
    """Calculate sector exposure for a portfolio, excluding bonds.

    Args:
        portfolio_name: Name of the portfolio (fuzzy-matched).
        portfolio_id: ID of the portfolio (exact match; takes precedence if both given).

    Returns:
        ExposureResult with both equity-renormalized and raw portfolio-weight views.
    """
    conn = get_connection()

    # Resolve portfolio
    pid, pname, err = _resolve_portfolio(conn, portfolio_name, portfolio_id)
    if err:
        return ExposureResult(error=err)

    assert pid is not None and pname is not None

    # Get holdings joined with securities and sectors
    query = """
        SELECT
            h.holding_id,
            s.security_id,
            s.asset_type,
            sec.sector_name,
            h.current_weight
        FROM holdings h
        JOIN securities s ON h.security_id = s.security_id
        LEFT JOIN sectors sec ON s.sector_id = sec.sector_id
        WHERE h.portfolio_id = ?
    """
    rows = conn.execute(query, (pid,)).fetchall()

    if not rows:
        return ExposureResult(
            portfolio_name=pname,
            portfolio_id=pid,
            error=f"No holdings found for portfolio '{pname}'.",
        )

    # Separate equities and bonds
    equity_weights: dict[str, float] = {}
    bonds_count = 0

    for _, _, asset_type, sector_name, weight in rows:
        if asset_type == "Bond":
            bonds_count += 1
            continue
        # Equity holding
        sector = sector_name or "Unknown"
        equity_weights[sector] = equity_weights.get(sector, 0.0) + (weight or 0.0)

    total_equity_weight = sum(equity_weights.values())

    # Edge case: all-bond portfolio
    if total_equity_weight == 0:
        return ExposureResult(
            portfolio_name=pname,
            portfolio_id=pid,
            bonds_excluded=bonds_count,
            error=(
                f"Portfolio '{pname}' has no equity holdings "
                f"(contains only {bonds_count} bond(s)). "
                f"Sector exposure is undefined for bond-only portfolios."
            ),
        )

    # Build both exposure views, sorted descending
    exposure_equity: dict[str, float] = {}
    exposure_portfolio: dict[str, float] = {}

    for sector, weight in equity_weights.items():
        exposure_equity[sector] = round((weight / total_equity_weight) * 100, 2)
        exposure_portfolio[sector] = round(weight * 100, 2)

    # Sort descending by equity exposure
    exposure_equity = dict(
        sorted(exposure_equity.items(), key=lambda x: x[1], reverse=True)
    )
    exposure_portfolio = dict(
        sorted(exposure_portfolio.items(), key=lambda x: x[1], reverse=True)
    )

    return ExposureResult(
        portfolio_name=pname,
        portfolio_id=pid,
        exposure_pct_of_equity=exposure_equity,
        exposure_pct_of_portfolio=exposure_portfolio,
        equity_weight_total=round(total_equity_weight * 100, 2),
        bonds_excluded=bonds_count,
    )

