"""
test_exposure.py – Tests for the sector exposure calculator.
"""

import pytest

from src.exposure_tool import ExposureResult, calculate_sector_exposure


class TestExposureCalculator:
    """Tests for sector exposure calculation."""

    def test_tech_innovation_fund_has_results(self):
        """Tech Innovation Fund (portfolio 3) should have equity exposure."""
        result = calculate_sector_exposure(portfolio_name="Tech Innovation Fund")
        assert not result.error
        assert result.portfolio_name == "Tech Innovation Fund"
        assert result.portfolio_id == 3
        assert len(result.exposure_pct_of_equity) > 0
        assert result.bonds_excluded == 0  # All equities

    def test_equity_exposure_sums_to_100(self):
        """Renormalized equity exposure should sum to ~100%."""
        result = calculate_sector_exposure(portfolio_name="Tech Innovation Fund")
        total = sum(result.exposure_pct_of_equity.values())
        assert abs(total - 100.0) < 0.1

    def test_fixed_income_plus_no_equity(self):
        """Fixed Income Plus (portfolio 8) has only bonds → error."""
        result = calculate_sector_exposure(portfolio_name="Fixed Income Plus")
        assert result.error
        assert "no equity" in result.error.lower()

    def test_total_bond_market_no_equity(self):
        """Total Bond Market Index Fund (portfolio 12) has only bonds → error."""
        result = calculate_sector_exposure(portfolio_name="Total Bond Market Index Fund")
        assert result.error
        assert "no equity" in result.error.lower()

    def test_conservative_income_mixed(self):
        """Conservative Income Fund (portfolio 2) is mixed – equity weight ~0.32."""
        result = calculate_sector_exposure(portfolio_name="Conservative Income Fund")
        assert not result.error
        assert result.equity_weight_total == 32.0
        assert result.bonds_excluded > 0

    def test_fuzzy_name_resolution(self):
        """'international equity' should resolve to 'International Equity Fund'."""
        result = calculate_sector_exposure(portfolio_name="international equity")
        assert not result.error
        assert result.portfolio_name == "International Equity Fund"

    def test_invalid_portfolio_name(self):
        """Non-existent portfolio should return error with available names."""
        result = calculate_sector_exposure(portfolio_name="Nonexistent Fund XYZ")
        assert result.error
        assert "Available" in result.error or "not found" in result.error.lower() or "Could not resolve" in result.error

    def test_portfolio_by_id(self):
        """Lookup by ID should work."""
        result = calculate_sector_exposure(portfolio_id=1)
        assert not result.error
        assert result.portfolio_name == "Growth Equity Fund"

    def test_sorted_descending(self):
        """Exposure sectors should be sorted descending by equity %."""
        result = calculate_sector_exposure(portfolio_name="Growth Equity Fund")
        values = list(result.exposure_pct_of_equity.values())
        assert values == sorted(values, reverse=True)

    def test_total_stock_market_weights(self):
        """Portfolio 11 weights sum to 0.98 – renormalization matters."""
        result = calculate_sector_exposure(portfolio_name="Total Stock Market Index Fund")
        assert not result.error
        assert result.equity_weight_total == 98.0
        # Renormalized should sum to 100
        total = sum(result.exposure_pct_of_equity.values())
        assert abs(total - 100.0) < 0.1

    def test_resolve_international_equity_token_match(self):
        """'international equity' must still resolve to 'International Equity Fund'."""
        result = calculate_sector_exposure(portfolio_name="international equity")
        assert not result.error
        assert result.portfolio_name == "International Equity Fund"

    def test_resolve_nonexistent_fund_returns_available(self):
        """'Nonexistent Fund' must return an error listing available portfolios."""
        result = calculate_sector_exposure(portfolio_name="Nonexistent Fund")
        assert result.error
        assert "Could not resolve" in result.error
        assert "Available portfolios" in result.error
        assert "Growth Equity Fund" in result.error

    def test_resolve_growth_fund_ambiguous_prompts_choice(self):
        """Ambiguous name 'Growth fund' must ask the user to choose among candidates, not pick one silently."""
        result = calculate_sector_exposure(portfolio_name="Growth fund")
        assert result.error
        assert "Ambiguous" in result.error
        assert "Did you mean" in result.error
        assert "Growth Equity Fund" in result.error


