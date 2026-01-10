"""Tests for async database module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from priority_lens.api.config import APIConfig
from priority_lens.api.database import Database, get_database, set_database


class TestDatabase:
    """Tests for Database class."""

    @pytest.fixture
    def config(self) -> APIConfig:
        """Create test config."""
        return APIConfig(
            database_url="postgresql+asyncpg://test:test@localhost:5432/test",
            db_pool_size=5,
            db_max_overflow=2,
            db_pool_recycle=1800,
            debug=False,
        )

    @pytest.fixture
    def database(self, config: APIConfig) -> Database:
        """Create database instance."""
        return Database(config)

    def test_init(self, config: APIConfig) -> None:
        """Test database initialization."""
        db = Database(config)
        assert db._config is config
        assert db._engine is None
        assert db._session_factory is None

    def test_engine_not_initialized(self, database: Database) -> None:
        """Test engine property raises when not initialized."""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            _ = database.engine

    def test_session_factory_not_initialized(self, database: Database) -> None:
        """Test session_factory property raises when not initialized."""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            _ = database.session_factory

    @pytest.mark.asyncio
    async def test_connect(self, database: Database) -> None:
        """Test database connection."""
        with (
            patch("priority_lens.api.database.create_async_engine") as mock_create_engine,
            patch("priority_lens.api.database.async_sessionmaker") as mock_sessionmaker,
        ):
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_factory = MagicMock()
            mock_sessionmaker.return_value = mock_factory

            await database.connect()

            mock_create_engine.assert_called_once()
            call_args = mock_create_engine.call_args
            assert "postgresql+asyncpg://test:test@localhost:5432/test" in str(call_args)

            mock_sessionmaker.assert_called_once()

            assert database._engine is mock_engine
            assert database._session_factory is mock_factory

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, database: Database) -> None:
        """Test that connect is idempotent."""
        with (
            patch("priority_lens.api.database.create_async_engine") as mock_create_engine,
            patch("priority_lens.api.database.async_sessionmaker") as mock_sessionmaker,
        ):
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = MagicMock()

            await database.connect()
            await database.connect()  # Second call should be no-op

            mock_create_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, database: Database) -> None:
        """Test database disconnection."""
        mock_engine = AsyncMock()
        database._engine = mock_engine
        database._session_factory = MagicMock()

        await database.disconnect()

        mock_engine.dispose.assert_called_once()
        assert database._engine is None
        assert database._session_factory is None

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, database: Database) -> None:
        """Test disconnect when not connected does nothing."""
        await database.disconnect()  # Should not raise
        assert database._engine is None

    @pytest.mark.asyncio
    async def test_session(self, database: Database) -> None:
        """Test session context manager."""
        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        database._session_factory = mock_factory

        async with database.session() as session:
            assert session is mock_session

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self, database: Database) -> None:
        """Test session rollback on exception."""
        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        database._session_factory = mock_factory

        with pytest.raises(ValueError):
            async with database.session():
                raise ValueError("Test error")

        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_connection_success(self, database: Database) -> None:
        """Test successful connection check."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()

        # Create a proper async context manager
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_cm
        database._engine = mock_engine

        result = await database.check_connection()

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_connection_not_connected(self, database: Database) -> None:
        """Test connection check when not connected."""
        result = await database.check_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_failure(self, database: Database) -> None:
        """Test connection check on failure."""
        mock_engine = AsyncMock()
        mock_engine.connect.side_effect = Exception("Connection failed")
        database._engine = mock_engine

        result = await database.check_connection()

        assert result is False

    def test_engine_property_after_connect(self, database: Database) -> None:
        """Test engine property after connection."""
        mock_engine = MagicMock()
        database._engine = mock_engine

        assert database.engine is mock_engine

    def test_session_factory_property_after_connect(self, database: Database) -> None:
        """Test session_factory property after connection."""
        mock_factory = MagicMock()
        database._session_factory = mock_factory

        assert database.session_factory is mock_factory


class TestGlobalDatabase:
    """Tests for global database functions."""

    def setup_method(self) -> None:
        """Reset global database before each test."""
        set_database(None)

    def test_get_database_not_initialized(self) -> None:
        """Test get_database raises when not initialized."""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            get_database()

    def test_set_and_get_database(self) -> None:
        """Test setting and getting global database."""
        mock_db = MagicMock(spec=Database)
        set_database(mock_db)

        result = get_database()
        assert result is mock_db

    def test_set_database_none(self) -> None:
        """Test clearing global database."""
        mock_db = MagicMock(spec=Database)
        set_database(mock_db)
        set_database(None)

        with pytest.raises(RuntimeError):
            get_database()
