"""Integration tests for Agent Runtime.

These tests verify the agent is correctly wired to the rest of the system:
- AgentRunner produces streaming events
- TurnService.invoke_agent() calls the agent and streams output
- Events are persisted to the database

Run with: uv run pytest tests/integration/test_agent_integration.py -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from priority_lens.agent.context import AgentContext
from priority_lens.agent.graph import AgentRunner
from priority_lens.models.canonical_event import EventType
from priority_lens.services.agent_streaming import AgentStreamingService, StreamingContext

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def get_test_db_url() -> str:
    """Get test database URL."""
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5433/test_priority_lens",
    )


def is_database_available() -> bool:
    """Check if the test database is available."""
    import asyncio

    async def check() -> bool:
        try:
            engine = create_async_engine(get_test_db_url(), echo=False)
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            return True
        except Exception:
            return False

    try:
        return asyncio.get_event_loop().run_until_complete(check())
    except RuntimeError:
        return asyncio.run(check())


# Mark for tests requiring database
requires_db = pytest.mark.skipif(not is_database_available(), reason="Test database not available")


@pytest.fixture(scope="module")
def engine():
    """Create async engine for tests."""
    return create_async_engine(get_test_db_url(), echo=False)


@pytest.fixture(scope="module")
def session_maker(engine):
    """Create session maker."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(session_maker) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for tests with rollback."""
    async with session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def agent_context() -> AgentContext:
    """Create an agent context for testing."""

    async def factory() -> AsyncGenerator[MagicMock, None]:
        yield MagicMock()

    return AgentContext(
        user_id=uuid4(),
        org_id=uuid4(),
        thread_id=uuid4(),
        session_factory=factory,
    )


@pytest.fixture
def streaming_context() -> StreamingContext:
    """Create a streaming context for testing."""
    return StreamingContext(
        thread_id=uuid4(),
        org_id=uuid4(),
        session_id=uuid4(),
        user_id=uuid4(),
        correlation_id=uuid4(),
    )


class TestAgentRunnerIntegration:
    """Tests for AgentRunner integration."""

    @pytest.mark.asyncio
    async def test_run_streaming_yields_events(self, agent_context: AgentContext) -> None:
        """Test that run_streaming yields proper AgentEvents."""
        from langchain_core.messages import AIMessage, HumanMessage

        # Mock the graph to return a simple response
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hello! How can I help you today?"),
                ]
            }
        )

        with patch("priority_lens.agent.graph.create_agent_graph", return_value=mock_graph):
            runner = AgentRunner(agent_context)
            events_gen = await runner.run_streaming("Hello")

            events = [event async for event in events_gen]

            assert len(events) == 1
            assert events[0].event_type == EventType.ASSISTANT_TEXT_FINAL
            assert "Hello" in events[0].payload["text"]

    @pytest.mark.asyncio
    async def test_run_streaming_handles_tool_calls(self, agent_context: AgentContext) -> None:
        """Test that run_streaming correctly handles tool calls."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        # Mock graph with tool call flow
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Show my inbox"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_priority_inbox",
                                "args": {"limit": 5},
                                "id": "call_abc123",
                            }
                        ],
                    ),
                    ToolMessage(content="[]", tool_call_id="call_abc123"),
                    AIMessage(content="Your inbox is empty."),
                ]
            }
        )

        with patch("priority_lens.agent.graph.create_agent_graph", return_value=mock_graph):
            runner = AgentRunner(agent_context)
            events_gen = await runner.run_streaming("Show my inbox")

            events = [event async for event in events_gen]

            # Should have: tool_call, tool_result, text_final
            assert len(events) == 3

            # Verify event types in order
            assert events[0].event_type == EventType.TOOL_CALL
            assert events[0].payload["tool_name"] == "get_priority_inbox"

            assert events[1].event_type == EventType.TOOL_RESULT
            assert events[1].payload["tool_call_id"] == "call_abc123"

            assert events[2].event_type == EventType.ASSISTANT_TEXT_FINAL
            assert "empty" in events[2].payload["text"].lower()


class TestAgentStreamingIntegration:
    """Tests for AgentStreaming integration with agent."""

    @requires_db
    @pytest.mark.asyncio
    async def test_stream_agent_output_persists_events(
        self, db_session: AsyncSession, streaming_context: StreamingContext
    ) -> None:
        """Test that streaming service persists agent events."""
        from priority_lens.services.agent_streaming import AgentEvent

        # Create mock event generator
        async def mock_events() -> AsyncGenerator[AgentEvent, None]:
            yield AgentEvent(
                event_type=EventType.ASSISTANT_TEXT_FINAL,
                payload={"text": "Test response"},
            )

        # Mock the event repository to avoid real DB writes in this test
        with patch.object(
            AgentStreamingService, "_emit_event", new_callable=AsyncMock
        ) as mock_emit:
            service = AgentStreamingService(db_session)
            events = await service.stream_agent_output(streaming_context, mock_events())

            # Should have: turn.agent.open, text_final, turn.agent.close
            assert len(events) == 3

            # Verify emit was called for each event
            assert mock_emit.call_count == 3

    @requires_db
    @pytest.mark.asyncio
    async def test_full_agent_to_streaming_flow(
        self,
        db_session: AsyncSession,
        agent_context: AgentContext,
        streaming_context: StreamingContext,
    ) -> None:
        """Test the complete flow: AgentRunner -> StreamingService."""
        from langchain_core.messages import AIMessage, HumanMessage

        # Mock graph
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ]
            }
        )

        with (
            patch("priority_lens.agent.graph.create_agent_graph", return_value=mock_graph),
            patch.object(AgentStreamingService, "_emit_event", new_callable=AsyncMock),
        ):
            # Create runner and get events
            runner = AgentRunner(agent_context)
            agent_events_gen = await runner.run_streaming("Hello")

            # Stream to service
            service = AgentStreamingService(db_session)
            emitted = await service.stream_agent_output(streaming_context, agent_events_gen)

            # Should have: turn.agent.open, text_final, turn.agent.close
            assert len(emitted) == 3

            # Verify event sequence
            assert emitted[0].event_type == EventType.TURN_AGENT_OPEN
            assert emitted[1].event_type == EventType.ASSISTANT_TEXT_FINAL
            assert emitted[2].event_type == EventType.TURN_AGENT_CLOSE


class TestTurnServiceAgentIntegration:
    """Tests for TurnService.invoke_agent integration."""

    @requires_db
    @pytest.mark.asyncio
    async def test_invoke_agent_returns_event_count(self, db_session: AsyncSession) -> None:
        """Test that invoke_agent returns the number of events emitted."""

        from langchain_core.messages import AIMessage, HumanMessage

        from priority_lens.services.turn_service import TurnService

        # Set up test data
        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()
        correlation_id = uuid4()

        # Mock session factory
        async def session_factory() -> AsyncGenerator[AsyncSession, None]:
            yield db_session

        # Mock graph response
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Test"),
                    AIMessage(content="Test response"),
                ]
            }
        )

        with (
            patch("priority_lens.agent.graph.create_agent_graph", return_value=mock_graph),
            patch.object(AgentStreamingService, "_emit_event", new_callable=AsyncMock),
        ):
            service = TurnService(db_session)
            count = await service.invoke_agent(
                thread_id=thread_id,
                org_id=org_id,
                user_id=user_id,
                session_id=session_id,
                correlation_id=correlation_id,
                user_message="Test",
                session_factory=lambda: session_factory(),
            )

            # Should emit: turn.agent.open, text_final, turn.agent.close
            assert count == 3
