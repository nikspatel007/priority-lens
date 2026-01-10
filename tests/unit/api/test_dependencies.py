"""Tests for FastAPI dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from priority_lens.api.database import Database, set_database
from priority_lens.api.dependencies import (
    get_db,
    get_db_session,
    get_org_id_optional,
    get_user_id_optional,
)


class TestGetDb:
    """Tests for get_db dependency."""

    def setup_method(self) -> None:
        """Reset global database before each test."""
        set_database(None)

    def test_returns_database(self) -> None:
        """Test that get_db returns the global database."""
        mock_db = MagicMock(spec=Database)
        set_database(mock_db)

        result = get_db()

        assert result is mock_db

    def test_raises_when_not_initialized(self) -> None:
        """Test that get_db raises when not initialized."""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            get_db()


class TestGetDbSession:
    """Tests for get_db_session dependency."""

    @pytest.mark.asyncio
    async def test_yields_session(self) -> None:
        """Test that get_db_session yields a session."""
        mock_session = AsyncMock()
        mock_db = AsyncMock(spec=Database)
        mock_db.session.return_value.__aenter__.return_value = mock_session
        mock_db.session.return_value.__aexit__.return_value = None

        async for session in get_db_session(mock_db):
            assert session is mock_session


class TestGetUserIdOptional:
    """Tests for get_user_id_optional dependency."""

    def test_returns_user_id(self) -> None:
        """Test that user ID is returned when provided."""
        result = get_user_id_optional("user-123")
        assert result == "user-123"

    def test_returns_none_when_not_provided(self) -> None:
        """Test that None is returned when no user ID."""
        result = get_user_id_optional(None)
        assert result is None


class TestGetOrgIdOptional:
    """Tests for get_org_id_optional dependency."""

    def test_returns_org_id(self) -> None:
        """Test that org ID is returned when provided."""
        result = get_org_id_optional("org-456")
        assert result == "org-456"

    def test_returns_none_when_not_provided(self) -> None:
        """Test that None is returned when no org ID."""
        result = get_org_id_optional(None)
        assert result is None
