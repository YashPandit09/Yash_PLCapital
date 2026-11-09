"""
db.py – In-memory SQLite database loader for Portfolio Analytics Agent.

Builds an in-memory SQLite database from the provided schema and CSV files.
Handles data quirks: NULL sector_id for bonds, quoted prices, empty strings → NULL.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Resolve project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _PROJECT_ROOT / "database_schema.sql"
_DATA_DIR = _PROJECT_ROOT / "data"

# Tables in dependency order matching the schema's foreign-key constraints.
_TABLE_LOAD_ORDER = [
    "sectors",
    "securities",
    "benchmarks",
    "portfolios",
    "holdings",
    "transactions",
    "historical_prices",
    "portfolio_performance",
    "risk_metrics",
]

# Singleton connection
_connection: Optional[sqlite3.Connection] = None


def _clean_value(value: str) -> Optional[str]:
    """Convert empty / whitespace-only strings to None (→ SQL NULL)."""
    if value is None:
        return None
    stripped = value.strip().strip('"').strip("'")
    return stripped if stripped != "" else None


def _load_csv(conn: sqlite3.Connection, table_name: str, csv_path: Path) -> int:
    """Load a single CSV file into the given table using csv.reader + executemany.

    Returns the number of rows inserted.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = [h.strip() for h in next(reader)]
        placeholders = ", ".join(["?"] * len(headers))
        col_names = ", ".join(headers)
        sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

        rows: list[tuple] = []
        for raw_row in reader:
            cleaned = tuple(_clean_value(v) for v in raw_row)
            rows.append(cleaned)

        conn.executemany(sql, rows)
        conn.commit()
        logger.info("Loaded %d rows into %s", len(rows), table_name)
        return len(rows)


def _build_database() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite DB, apply schema, then load CSVs."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    # Apply schema
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    logger.info("Schema applied from %s", _SCHEMA_PATH)

    # Load CSVs in dependency order
    for table_name in _TABLE_LOAD_ORDER:
        csv_path = _DATA_DIR / f"{table_name}.csv"
        if csv_path.exists():
            _load_csv(conn, table_name, csv_path)
        else:
            logger.warning("CSV not found for table %s: %s", table_name, csv_path)

    return conn


def get_connection() -> sqlite3.Connection:
    """Return the singleton in-memory database connection, building it on first call."""
    global _connection
    if _connection is None:
        _connection = _build_database()
    return _connection


def get_schema_description() -> str:
    """Return a compact text description of the database schema for LLM context.

    Includes table names, columns with types, and sample distinct values
    for key categorical columns to help the LLM generate accurate SQL.
    """
    conn = get_connection()
    lines: list[str] = ["=== DATABASE SCHEMA ===\n"]

    # Get all tables
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    for (table_name,) in tables:
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_descs = []
        for col in cols:
            # col: (cid, name, type, notnull, default_value, pk)
            name, dtype, notnull, pk = col[1], col[2], col[3], col[5]
            parts = [f"{name} {dtype}"]
            if pk:
                parts.append("PK")
            if notnull:
                parts.append("NOT NULL")
            col_descs.append(" ".join(parts))

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        lines.append(f"TABLE {table_name} ({row_count} rows):")
        lines.append(f"  Columns: {', '.join(col_descs)}")

    # Add distinct-value hints for key columns
    lines.append("\n=== KEY VALUE HINTS ===\n")

    hints: list[tuple[str, str]] = [
        ("portfolios", "status"),
        ("portfolios", "target_risk_level"),
        ("portfolios", "strategy_type"),
        ("securities", "asset_type"),
        ("securities", "currency"),
        ("securities", "exchange"),
        ("transactions", "transaction_type"),
    ]
    for table, col in hints:
        try:
            vals = conn.execute(
                f"SELECT DISTINCT {col} FROM {table} ORDER BY {col}"
            ).fetchall()
            val_strs = [str(v[0]) for v in vals if v[0] is not None]
            lines.append(f"{table}.{col} ∈ {{{', '.join(val_strs)}}}")
        except sqlite3.Error:
            pass

    # Portfolio names
    pnames = conn.execute(
        "SELECT portfolio_name FROM portfolios ORDER BY portfolio_id"
    ).fetchall()
    lines.append(
        f"\nAll portfolio names: {', '.join(repr(p[0]) for p in pnames)}"
    )

    # Sector names
    snames = conn.execute(
        "SELECT sector_name FROM sectors ORDER BY sector_id"
    ).fetchall()
    lines.append(
        f"All sector names: {', '.join(repr(s[0]) for s in snames)}"
    )

    # Benchmark names
    bnames = conn.execute(
        "SELECT benchmark_name FROM benchmarks ORDER BY benchmark_id"
    ).fetchall()
    lines.append(
        f"All benchmark names: {', '.join(repr(b[0]) for b in bnames)}"
    )

    return "\n".join(lines)


def reset_connection() -> None:
    """Reset the singleton connection (useful for testing)."""
    global _connection
    if _connection is not None:
        _connection.close()
    _connection = None

