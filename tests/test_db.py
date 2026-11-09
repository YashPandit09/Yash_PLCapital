"""
test_db.py – Tests for the database loader.
"""

import sqlite3

import pytest

from src.db import get_connection, get_schema_description, reset_connection


@pytest.fixture(autouse=True)
def _fresh_db():
    """Ensure each test gets the DB initialised (singleton is fine for reads)."""
    yield
    # Don't reset between tests – singleton is read-only in these tests


class TestDatabaseLoader:
    """Tests for database loading and schema."""

    def test_connection_returns_sqlite_connection(self):
        conn = get_connection()
        assert isinstance(conn, sqlite3.Connection)

    def test_tables_exist(self):
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {t[0] for t in tables}
        expected = {
            "sectors", "securities", "benchmarks", "portfolios",
            "holdings", "transactions", "historical_prices",
            "portfolio_performance", "risk_metrics",
        }
        assert expected.issubset(table_names)

    def test_sectors_loaded(self):
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM sectors").fetchone()[0]
        assert count == 10

    def test_securities_loaded(self):
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        assert count == 30

    def test_portfolios_loaded(self):
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
        assert count == 13

    def test_holdings_loaded(self):
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
        assert count == 95

    def test_bonds_have_null_sector(self):
        """Bonds in securities should have NULL sector_id."""
        conn = get_connection()
        bonds = conn.execute(
            "SELECT security_id, symbol, sector_id FROM securities WHERE asset_type = 'Bond'"
        ).fetchall()
        assert len(bonds) == 5
        for _, _, sector_id in bonds:
            assert sector_id is None

    def test_bond_prices_parsed_correctly(self):
        """Bond prices should be numeric despite quoting in CSV."""
        conn = get_connection()
        price = conn.execute(
            "SELECT current_price FROM securities WHERE symbol = 'UST-10Y'"
        ).fetchone()[0]
        assert isinstance(price, (int, float))
        assert abs(price - 102.50) < 0.01

    def test_foreign_keys_enforced(self):
        """Foreign key constraints should be active."""
        conn = get_connection()
        fk_status = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk_status == 1

    def test_schema_description_not_empty(self):
        desc = get_schema_description()
        assert len(desc) > 100
        assert "portfolios" in desc
        assert "sectors" in desc

