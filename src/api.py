"""
api.py – Production-ready FastAPI REST API for Portfolio Analytics Agent.

Usage:
    uvicorn src.api:app --port 8000
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.agent import AgentResult, PortfolioAgent
from src.db import get_connection
from src.exposure_tool import ExposureResult, calculate_sector_exposure

# Initialize application
app = FastAPI(
    title="Portfolio Analytics Agent API",
    description="REST API for automated natural-language querying and sector exposure analysis.",
    version="1.0.0",
)

# Shared agent singleton
_agent = PortfolioAgent()


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    database: str = "in-memory-sqlite"


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about portfolio analytics.",
        json_schema_extra={"example": "What are the sector exposures for the Tech Innovation Fund?"},
    )


class QueryResponse(BaseModel):
    question: str
    tool_used: str
    tool_input: Optional[Any] = None
    answer: str
    error: Optional[str] = None
    latency_ms: float


class PortfolioSummary(BaseModel):
    portfolio_id: int
    portfolio_name: str
    target_risk_level: Optional[str]
    total_aum: Optional[float]
    strategy_type: Optional[str]
    status: Optional[str]


class ExposureResponse(BaseModel):
    portfolio_name: str
    portfolio_id: int
    exposure_pct_of_equity: dict[str, float]
    exposure_pct_of_portfolio: dict[str, float]
    equity_weight_total: float
    bonds_excluded: int
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Check system health and database connectivity."""
    return HealthResponse()


@app.get("/portfolios", response_model=list[PortfolioSummary], tags=["Portfolios"])
def list_portfolios(status: Optional[str] = Query(None, description="Filter by status (e.g. Active, Passive)")) -> list[PortfolioSummary]:
    """Retrieve all available portfolios."""
    conn = get_connection()
    if status:
        rows = conn.execute(
            """
            SELECT portfolio_id, portfolio_name, target_risk_level, total_aum, strategy_type, status
            FROM portfolios
            WHERE LOWER(status) = LOWER(?)
            ORDER BY portfolio_id
            """,
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT portfolio_id, portfolio_name, target_risk_level, total_aum, strategy_type, status
            FROM portfolios
            ORDER BY portfolio_id
            """
        ).fetchall()

    return [
        PortfolioSummary(
            portfolio_id=r[0],
            portfolio_name=r[1],
            target_risk_level=r[2],
            total_aum=r[3],
            strategy_type=r[4],
            status=r[5],
        )
        for r in rows
    ]


@app.get("/exposure/{portfolio_name}", response_model=ExposureResponse, tags=["Analytics"])
def get_sector_exposure(portfolio_name: str) -> ExposureResponse:
    """Calculate deterministic sector exposure for a specific portfolio."""
    result: ExposureResult = calculate_sector_exposure(portfolio_name=portfolio_name)
    if result.error and "not found" in result.error.lower():
        raise HTTPException(status_code=404, detail=result.error)

    return ExposureResponse(
        portfolio_name=result.portfolio_name,
        portfolio_id=result.portfolio_id,
        exposure_pct_of_equity=result.exposure_pct_of_equity,
        exposure_pct_of_portfolio=result.exposure_pct_of_portfolio,
        equity_weight_total=result.equity_weight_total,
        bonds_excluded=result.bonds_excluded,
        error=result.error or None,
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def answer_query(request: QueryRequest) -> QueryResponse:
    """Submit a natural language question to the Portfolio Agent."""
    result: AgentResult = _agent.answer_question(request.question)
    return QueryResponse(
        question=result.question,
        tool_used=result.tool_used,
        tool_input=result.tool_input,
        answer=result.answer,
        error=result.error or None,
        latency_ms=result.latency_ms,
    )
