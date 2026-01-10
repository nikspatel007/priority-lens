"""Agent context for tool execution."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Type alias for session factory
SessionFactory = Callable[[], AsyncGenerator["AsyncSession", None]]


@dataclass
class AgentContext:
    """Context for agent tool execution.

    Provides access to user identity and database sessions.
    """

    user_id: UUID
    org_id: UUID
    thread_id: UUID
    session_factory: SessionFactory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session.

        Yields:
            AsyncSession instance.
        """
        async for sess in self.session_factory():
            yield sess
            return
