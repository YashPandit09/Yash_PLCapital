"""
test_api.py – Tests for FastAPI REST API endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from src.api import app

client = TestClient(app)


class TestPortfolioAPI:
    """Tests for the REST API endpoints."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "in-memory-sqlite"

    def test_list_portfolios(self):
        response = client.get("/portfolios")
        assert response.status_code == 200
        portfolios = response.json()
        assert len(portfolios) == 13
        assert portfolios[0]["portfolio_name"] == "Growth Equity Fund"

    def test_list_portfolios_filtered(self):
        response = client.get("/portfolios?status=Active")
        assert response.status_code == 200
        active = response.json()
        assert len(active) == 10
        for p in active:
            assert p["status"] == "Active"

    def test_exposure_endpoint(self):
        response = client.get("/exposure/Tech%20Innovation%20Fund")
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_name"] == "Tech Innovation Fund"
        assert "Technology" in data["exposure_pct_of_equity"]
        assert data["exposure_pct_of_equity"]["Technology"] == 100.0

    def test_exposure_fuzzy_match(self):
        response = client.get("/exposure/international%20equity")
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_name"] == "International Equity Fund"
        assert len(data["exposure_pct_of_equity"]) > 0

    def test_query_endpoint_exposure(self):
        response = client.post(
            "/query",
            json={"question": "What are the sector exposures for the Tech Innovation Fund?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tool_used"] == "exposure_calculator"
        assert "Technology" in data["answer"]

