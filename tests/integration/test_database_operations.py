"""Integration tests for database operations.

These tests require a running PostgreSQL database. They are skipped if
the database is not available.

Run with: uv run pytest tests/integration/ -v
"""

from __future__ import annotations

import os

import psycopg2
import pytest

from rl_emails.core.db import fetch_count, fetch_one_value, get_connection, get_cursor


def is_database_available() -> bool:
    """Check if the test database is available."""
    db_url = os.environ.get(
        "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/test_rl_emails"
    )
    try:
        conn = psycopg2.connect(db_url)
        conn.close()
        return True
    except Exception:
        return False


# Skip all tests in this module if database is not available
pytestmark = pytest.mark.skipif(not is_database_available(), reason="Test database not available")


@pytest.fixture(scope="module")
def test_db_url() -> str:
    """Get test database URL."""
    return os.environ.get(
        "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/test_rl_emails"
    )


@pytest.fixture
def db_connection(test_db_url: str) -> psycopg2.extensions.connection:
    """Create a database connection for tests."""
    conn = psycopg2.connect(test_db_url)
    yield conn
    conn.rollback()
    conn.close()


class TestDatabaseConnection:
    """Integration tests for database connection utilities."""

    def test_get_connection_works(self, test_db_url: str) -> None:
        """Test that get_connection successfully connects to database."""
        with get_connection(test_db_url) as conn:
            assert conn is not None
            # Verify we can execute a simple query
            with get_cursor(conn) as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                assert result is not None
                assert result[0] == 1

    def test_get_cursor_works(self, db_connection: psycopg2.extensions.connection) -> None:
        """Test that get_cursor returns a working cursor."""
        with get_cursor(db_connection) as cur:
            cur.execute("SELECT version()")
            result = cur.fetchone()
            assert result is not None
            assert "PostgreSQL" in result[0]

    def test_fetch_count_with_real_query(
        self, db_connection: psycopg2.extensions.connection
    ) -> None:
        """Test fetch_count with actual database query."""
        with get_cursor(db_connection) as cur:
            # Create a temp table, insert data, count it
            cur.execute("CREATE TEMP TABLE test_count (id serial)")
            cur.execute("INSERT INTO test_count DEFAULT VALUES")
            cur.execute("INSERT INTO test_count DEFAULT VALUES")
            cur.execute("SELECT COUNT(*) FROM test_count")
            count = fetch_count(cur)
            assert count == 2
            assert isinstance(count, int)

    def test_fetch_one_value_with_real_query(
        self, db_connection: psycopg2.extensions.connection
    ) -> None:
        """Test fetch_one_value with actual database query."""
        with get_cursor(db_connection) as cur:
            cur.execute("SELECT 'hello_world'")
            value = fetch_one_value(cur)
            assert value == "hello_world"

    def test_fetch_one_value_returns_none_for_empty(
        self, db_connection: psycopg2.extensions.connection
    ) -> None:
        """Test fetch_one_value returns None for empty result."""
        with get_cursor(db_connection) as cur:
            cur.execute("CREATE TEMP TABLE test_empty (id int)")
            cur.execute("SELECT id FROM test_empty")
            value = fetch_one_value(cur)
            assert value is None


class TestDatabaseTransactions:
    """Integration tests for transaction handling."""

    def test_connection_rolls_back_on_exception(self, test_db_url: str) -> None:
        """Test that connection context manager doesn't auto-commit on error."""
        table_created = False

        try:
            with get_connection(test_db_url) as conn:
                with get_cursor(conn) as cur:
                    cur.execute("CREATE TABLE test_rollback_table (id int)")
                    table_created = True
                    raise ValueError("Force rollback")
        except ValueError:
            pass

        # Table should NOT exist because we didn't commit
        if table_created:
            with get_connection(test_db_url) as conn:
                with get_cursor(conn) as cur:
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'test_rollback_table'
                        )
                    """
                    )
                    exists = fetch_one_value(cur)
                    assert exists is False, "Table should not exist after rollback"

    def test_explicit_commit_persists_data(self, test_db_url: str) -> None:
        """Test that explicit commit persists data."""
        # Use a unique table name to avoid conflicts
        import uuid

        table_name = f"test_commit_{uuid.uuid4().hex[:8]}"

        try:
            with get_connection(test_db_url) as conn:
                with get_cursor(conn) as cur:
                    cur.execute(f"CREATE TABLE {table_name} (id int)")
                    cur.execute(f"INSERT INTO {table_name} VALUES (1)")
                    conn.commit()

            # Verify data persisted
            with get_connection(test_db_url) as conn:
                with get_cursor(conn) as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = fetch_count(cur)
                    assert count == 1
        finally:
            # Cleanup
            with get_connection(test_db_url) as conn:
                with get_cursor(conn) as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                    conn.commit()
