"""
test_sql_validator.py – Tests for SQL safety validation.
"""

import pytest

from src.sql_tool import validate_sql


class TestSQLValidator:
    """Tests for the SQL safety validator."""

    def test_valid_select(self):
        ok, err = validate_sql("SELECT COUNT(*) FROM portfolios")
        assert ok
        assert err == ""

    def test_valid_with_cte(self):
        ok, err = validate_sql(
            "WITH cte AS (SELECT 1) SELECT * FROM cte"
        )
        assert ok

    def test_reject_insert(self):
        ok, err = validate_sql("INSERT INTO portfolios VALUES (99, 'X')")
        assert not ok
        assert len(err) > 0

    def test_reject_insert_inside_cte(self):
        # Even if someone tries to hide it after a SELECT
        ok, err = validate_sql("SELECT 1; INSERT INTO portfolios VALUES (99, 'X')")
        assert not ok

    def test_reject_update(self):
        ok, err = validate_sql("UPDATE portfolios SET status='Inactive'")
        assert not ok

    def test_reject_delete(self):
        ok, err = validate_sql("DELETE FROM portfolios WHERE 1=1")
        assert not ok

    def test_reject_drop(self):
        ok, err = validate_sql("DROP TABLE portfolios")
        assert not ok

    def test_reject_multiple_statements(self):
        ok, err = validate_sql(
            "SELECT 1; DROP TABLE portfolios"
        )
        assert not ok

    def test_reject_pragma(self):
        ok, err = validate_sql("PRAGMA table_info(portfolios)")
        assert not ok

    def test_reject_attach(self):
        ok, err = validate_sql("ATTACH DATABASE ':memory:' AS hack")
        assert not ok

    def test_empty_sql(self):
        ok, err = validate_sql("")
        assert not ok

    def test_trailing_semicolon_ok(self):
        ok, err = validate_sql("SELECT 1;")
        assert ok

