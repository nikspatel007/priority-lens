"""Tests for priority_lens.core.db."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from priority_lens.core.db import (
    fetch_count,
    fetch_one_value,
    get_connection,
    get_cursor,
    get_database_url,
)


class TestGetDatabaseUrl:
    """Tests for get_database_url function."""

    def test_returns_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that it returns DATABASE_URL from environment."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/testdb")
        assert get_database_url() == "postgresql://localhost/testdb"

    def test_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that it raises ValueError when DATABASE_URL not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="DATABASE_URL environment variable not set"):
            get_database_url()


class TestGetConnection:
    """Tests for get_connection context manager."""

    @patch("priority_lens.core.db.psycopg2.connect")
    def test_yields_connection(self, mock_connect: MagicMock) -> None:
        """Test that it yields a connection and closes it."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with get_connection("postgresql://test") as conn:
            assert conn == mock_conn

        mock_connect.assert_called_once_with("postgresql://test")
        mock_conn.close.assert_called_once()

    @patch("priority_lens.core.db.psycopg2.connect")
    @patch("priority_lens.core.db.get_database_url")
    def test_uses_env_url_when_none(self, mock_get_url: MagicMock, mock_connect: MagicMock) -> None:
        """Test that it uses get_database_url when db_url is None."""
        mock_get_url.return_value = "postgresql://env_url"
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with get_connection() as conn:
            assert conn == mock_conn

        mock_get_url.assert_called_once()
        mock_connect.assert_called_once_with("postgresql://env_url")

    @patch("priority_lens.core.db.psycopg2.connect")
    def test_closes_on_exception(self, mock_connect: MagicMock) -> None:
        """Test that connection is closed even on exception."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with pytest.raises(RuntimeError):
            with get_connection("postgresql://test") as _conn:
                raise RuntimeError("Test error")

        mock_conn.close.assert_called_once()


class TestGetCursor:
    """Tests for get_cursor context manager."""

    def test_yields_cursor(self) -> None:
        """Test that it yields a cursor and closes it."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with get_cursor(mock_conn) as cur:
            assert cur == mock_cursor

        mock_conn.cursor.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_closes_on_exception(self) -> None:
        """Test that cursor is closed even on exception."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with pytest.raises(RuntimeError):
            with get_cursor(mock_conn) as _cur:
                raise RuntimeError("Test error")

        mock_cursor.close.assert_called_once()


class TestFetchOneValue:
    """Tests for fetch_one_value function."""

    def test_returns_first_column(self) -> None:
        """Test that it returns first column of first row."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("value", "other")

        result = fetch_one_value(mock_cursor)
        assert result == "value"

    def test_returns_default_when_none(self) -> None:
        """Test that it returns default when fetchone returns None."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        result = fetch_one_value(mock_cursor, default="default_value")
        assert result == "default_value"

    def test_returns_none_as_default(self) -> None:
        """Test that default is None when not specified."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        result = fetch_one_value(mock_cursor)
        assert result is None


class TestFetchCount:
    """Tests for fetch_count function."""

    def test_returns_count_as_int(self) -> None:
        """Test that it returns count as integer."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (42,)

        result = fetch_count(mock_cursor)
        assert result == 42
        assert isinstance(result, int)

    def test_returns_zero_when_none(self) -> None:
        """Test that it returns 0 when fetchone returns None."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        result = fetch_count(mock_cursor)
        assert result == 0

    def test_converts_to_int(self) -> None:
        """Test that it converts result to int."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100.0,)

        result = fetch_count(mock_cursor)
        assert result == 100
        assert isinstance(result, int)
