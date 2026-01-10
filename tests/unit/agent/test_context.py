"""Tests for agent context."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from priority_lens.agent.context import AgentContext


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
async def mock_session_factory(mock_session: MagicMock) -> AsyncGenerator[MagicMock, None]:
    """Create a mock session factory."""

    async def factory() -> AsyncGenerator[MagicMock, None]:
        yield mock_session

    yield factory  # type: ignore[misc]


class TestAgentContext:
    """Tests for AgentContext."""

    def test_context_creation(self) -> None:
        """Test creating an agent context."""
        user_id = uuid4()
        org_id = uuid4()
        thread_id = uuid4()

        async def factory() -> AsyncGenerator[MagicMock, None]:
            yield MagicMock()

        ctx = AgentContext(
            user_id=user_id,
            org_id=org_id,
            thread_id=thread_id,
            session_factory=factory,
        )

        assert ctx.user_id == user_id
        assert ctx.org_id == org_id
        assert ctx.thread_id == thread_id
        assert ctx.session_factory == factory

    @pytest.mark.asyncio
    async def test_session_context_manager(self, mock_session: MagicMock) -> None:
        """Test the session context manager yields a session."""

        async def factory() -> AsyncGenerator[MagicMock, None]:
            yield mock_session

        ctx = AgentContext(
            user_id=uuid4(),
            org_id=uuid4(),
            thread_id=uuid4(),
            session_factory=factory,
        )

        async with ctx.session() as session:
            assert session is mock_session
