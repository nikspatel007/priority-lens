"""Async database engine and session management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

if TYPE_CHECKING:
    from rl_emails.api.config import APIConfig


class Database:
    """Async database manager with connection pooling."""

    def __init__(self, config: APIConfig) -> None:
        """Initialize database with configuration.

        Args:
            config: API configuration with database settings.
        """
        self._config = config
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Get the async engine, raising if not initialized."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory, raising if not initialized."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._session_factory

    async def connect(self) -> None:
        """Create the async engine and session factory."""
        if self._engine is not None:
            return

        self._engine = create_async_engine(
            self._config.database_url,
            pool_size=self._config.db_pool_size,
            max_overflow=self._config.db_max_overflow,
            pool_recycle=self._config.db_pool_recycle,
            pool_pre_ping=True,
            echo=self._config.debug,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    async def disconnect(self) -> None:
        """Close the async engine and clean up connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create a new async session context.

        Yields:
            AsyncSession: Database session that auto-closes on exit.

        Raises:
            RuntimeError: If database is not initialized.
        """
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def check_connection(self) -> bool:
        """Check if database connection is healthy.

        Returns:
            True if connection is healthy, False otherwise.
        """
        if self._engine is None:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


# Global database instance (set during app startup)
_database: Database | None = None


def get_database() -> Database:
    """Get the global database instance.

    Returns:
        The global Database instance.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _database is None:
        raise RuntimeError("Database not initialized. App startup incomplete.")
    return _database


def set_database(db: Database | None) -> None:
    """Set the global database instance.

    Args:
        db: Database instance or None to clear.
    """
    global _database
    _database = db


# Re-export for type hints - text is used directly above
__all__: list[Any] = ["Database", "get_database", "set_database"]
