"""Database connection utilities."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
import psycopg2.extensions


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable not set")
    return url


@contextmanager
def get_connection(
    db_url: str | None = None,
) -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager for database connections.

    Args:
        db_url: Database URL. If None, uses DATABASE_URL env var.

    Yields:
        Database connection that auto-closes on exit.
    """
    url = db_url or get_database_url()
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_cursor(
    conn: psycopg2.extensions.connection,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager for database cursors.

    Args:
        conn: Database connection.

    Yields:
        Cursor that auto-closes on exit.
    """
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def fetch_one_value(cur: psycopg2.extensions.cursor, default: Any = None) -> Any:
    """Safely fetch a single value from cursor.

    Args:
        cur: Database cursor after executing query.
        default: Value to return if no row found.

    Returns:
        First column of first row, or default if no results.
    """
    row = cur.fetchone()
    return row[0] if row else default


def fetch_count(cur: psycopg2.extensions.cursor) -> int:
    """Safely fetch a count value from cursor.

    Args:
        cur: Database cursor after executing COUNT query.

    Returns:
        Count value as int, or 0 if no results.
    """
    row = cur.fetchone()
    return int(row[0]) if row else 0
