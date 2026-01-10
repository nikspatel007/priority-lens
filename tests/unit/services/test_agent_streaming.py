"""Tests for agent streaming service."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from priority_lens.models.canonical_event import EventActor, EventType
from priority_lens.services.agent_streaming import (
    AgentCancellationError,
    AgentEvent,
    AgentStreamingService,
    AgentStreamingState,
    StreamingContext,
)


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def mock_livekit_service() -> MagicMock:
    """Create a mock LiveKit service."""
    service = MagicMock()
    service.get_room_name.return_value = "pl-thread-test"
    return service


@pytest.fixture
def streaming_context() -> StreamingContext:
    """Create a streaming context for tests."""
    return StreamingContext(
        thread_id=uuid4(),
        org_id=uuid4(),
        session_id=uuid4(),
        user_id=uuid4(),
        correlation_id=uuid4(),
        livekit_room="pl-thread-test",
    )


class TestStreamingContext:
    """Tests for StreamingContext dataclass."""

    def test_streaming_context_creation(self) -> None:
        """Test creating a streaming context."""
        thread_id = uuid4()
        org_id = uuid4()
        session_id = uuid4()
        user_id = uuid4()
        correlation_id = uuid4()

        ctx = StreamingContext(
            thread_id=thread_id,
            org_id=org_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
        )

        assert ctx.thread_id == thread_id
        assert ctx.org_id == org_id
        assert ctx.session_id == session_id
        assert ctx.user_id == user_id
        assert ctx.correlation_id == correlation_id
        assert ctx.livekit_room is None

    def test_streaming_context_with_livekit_room(self) -> None:
        """Test creating a streaming context with LiveKit room."""
        ctx = StreamingContext(
            thread_id=uuid4(),
            org_id=uuid4(),
            session_id=uuid4(),
            user_id=uuid4(),
            correlation_id=uuid4(),
            livekit_room="pl-thread-abc123",
        )

        assert ctx.livekit_room == "pl-thread-abc123"


class TestAgentEvent:
    """Tests for AgentEvent dataclass."""

    def test_agent_event_creation(self) -> None:
        """Test creating an agent event."""
        event = AgentEvent(
            event_type=EventType.ASSISTANT_TEXT_DELTA,
            payload={"text": "Hello"},
        )

        assert event.event_type == EventType.ASSISTANT_TEXT_DELTA
        assert event.payload == {"text": "Hello"}
        assert event.actor == EventActor.AGENT

    def test_agent_event_with_custom_actor(self) -> None:
        """Test creating an agent event with custom actor."""
        event = AgentEvent(
            event_type=EventType.SYSTEM_ERROR,
            payload={"error": "Something went wrong"},
            actor=EventActor.SYSTEM,
        )

        assert event.actor == EventActor.SYSTEM


class TestAgentStreamingState:
    """Tests for AgentStreamingState dataclass."""

    def test_initial_state(self) -> None:
        """Test initial streaming state."""
        state = AgentStreamingState()

        assert state.is_cancelled is False
        assert state.events_emitted == 0
        assert state.turn_open is False

    def test_state_modification(self) -> None:
        """Test modifying streaming state."""
        state = AgentStreamingState()

        state.is_cancelled = True
        state.events_emitted = 5
        state.turn_open = True

        assert state.is_cancelled is True
        assert state.events_emitted == 5
        assert state.turn_open is True


class TestAgentStreamingService:
    """Tests for AgentStreamingService class."""

    @pytest.mark.asyncio
    async def test_stream_agent_output_success(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test successful streaming of agent output."""

        async def mock_agent_events() -> AsyncGenerator[AgentEvent, None]:
            yield AgentEvent(
                event_type=EventType.ASSISTANT_TEXT_DELTA,
                payload={"delta": "Hello"},
            )
            yield AgentEvent(
                event_type=EventType.ASSISTANT_TEXT_FINAL,
                payload={"text": "Hello, world!"},
            )

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)
            events = await service.stream_agent_output(streaming_context, mock_agent_events())

        # Should have: turn.agent.open + 2 content events + turn.agent.close
        assert len(events) == 4

        # First event is turn.agent.open
        assert events[0].event_type == EventType.TURN_AGENT_OPEN
        assert events[0].payload["correlation_id"] == str(streaming_context.correlation_id)

        # Content events
        assert events[1].event_type == EventType.ASSISTANT_TEXT_DELTA
        assert events[2].event_type == EventType.ASSISTANT_TEXT_FINAL

        # Last event is turn.agent.close
        assert events[3].event_type == EventType.TURN_AGENT_CLOSE
        assert events[3].payload["reason"] == "complete"
        assert events[3].payload["events_emitted"] == 2

    @pytest.mark.asyncio
    async def test_stream_agent_output_empty(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test streaming with no content events."""

        async def mock_agent_events() -> AsyncGenerator[AgentEvent, None]:
            # Empty generator - no events
            return
            yield  # Make it a generator

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)
            events = await service.stream_agent_output(streaming_context, mock_agent_events())

        # Should have: turn.agent.open + turn.agent.close
        assert len(events) == 2
        assert events[0].event_type == EventType.TURN_AGENT_OPEN
        assert events[1].event_type == EventType.TURN_AGENT_CLOSE
        assert events[1].payload["events_emitted"] == 0

    @pytest.mark.asyncio
    async def test_stream_agent_output_with_cancellation(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test streaming with cancellation."""
        events_generated = 0

        async def mock_agent_events() -> AsyncGenerator[AgentEvent, None]:
            nonlocal events_generated
            for i in range(10):
                events_generated += 1
                yield AgentEvent(
                    event_type=EventType.ASSISTANT_TEXT_DELTA,
                    payload={"delta": f"chunk{i}"},
                )

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)

            # Start streaming and cancel after first event
            async def cancel_after_first() -> list[AgentEvent]:
                events: list[AgentEvent] = []
                gen = mock_agent_events()

                async for event in gen:
                    # Process first event
                    events.append(event)
                    # Cancel after first event
                    service._active_sessions[streaming_context.correlation_id] = (
                        AgentStreamingState(is_cancelled=True, turn_open=True)
                    )
                    break

                return events

            # Test that cancellation is detected
            service._active_sessions[streaming_context.correlation_id] = AgentStreamingState()

            with pytest.raises(AgentCancellationError):
                await service.stream_agent_output(
                    streaming_context,
                    self._create_cancellable_generator(service, streaming_context),
                )

    async def _create_cancellable_generator(
        self,
        service: AgentStreamingService,
        ctx: StreamingContext,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Create a generator that triggers cancellation."""
        yield AgentEvent(
            event_type=EventType.ASSISTANT_TEXT_DELTA,
            payload={"delta": "first"},
        )
        # Mark session as cancelled
        if ctx.correlation_id in service._active_sessions:
            service._active_sessions[ctx.correlation_id].is_cancelled = True
        yield AgentEvent(
            event_type=EventType.ASSISTANT_TEXT_DELTA,
            payload={"delta": "should not be emitted"},
        )

    @pytest.mark.asyncio
    async def test_stream_agent_output_with_error(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test streaming with error during processing."""

        async def mock_agent_events_with_error() -> AsyncGenerator[AgentEvent, None]:
            yield AgentEvent(
                event_type=EventType.ASSISTANT_TEXT_DELTA,
                payload={"delta": "starting"},
            )
            raise ValueError("Test error")

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)

            with pytest.raises(ValueError, match="Test error"):
                await service.stream_agent_output(streaming_context, mock_agent_events_with_error())

            # Verify error close event was emitted
            calls = mock_repo.append_event_raw.call_args_list

            # Should have: turn.agent.open, delta, turn.agent.close (error)
            assert len(calls) == 3

            # Last call should be turn.agent.close with error reason
            last_call_kwargs = calls[-1].kwargs
            assert last_call_kwargs["event_type"] == EventType.TURN_AGENT_CLOSE
            assert last_call_kwargs["payload"]["reason"] == "error"
            assert "Test error" in last_call_kwargs["payload"]["error"]

    @pytest.mark.asyncio
    async def test_cancel_agent_success(
        self,
        mock_session: MagicMock,
    ) -> None:
        """Test cancelling an active agent session."""
        correlation_id = uuid4()

        service = AgentStreamingService(mock_session)
        service._active_sessions[correlation_id] = AgentStreamingState()

        result = await service.cancel_agent(correlation_id)

        assert result is True
        assert service._active_sessions[correlation_id].is_cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_agent_not_found(
        self,
        mock_session: MagicMock,
    ) -> None:
        """Test cancelling a non-existent agent session."""
        correlation_id = uuid4()

        service = AgentStreamingService(mock_session)
        result = await service.cancel_agent(correlation_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_emit_system_cancel_event(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test emitting a system cancel event."""
        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)
            await service.emit_system_cancel_event(streaming_context, "user_request")

            mock_repo.append_event_raw.assert_called_once()
            call_kwargs = mock_repo.append_event_raw.call_args.kwargs
            assert call_kwargs["event_type"] == EventType.SYSTEM_CANCEL
            assert call_kwargs["payload"]["reason"] == "user_request"
            assert call_kwargs["actor"] == EventActor.SYSTEM

    def test_get_active_session_count(
        self,
        mock_session: MagicMock,
    ) -> None:
        """Test getting active session count."""
        service = AgentStreamingService(mock_session)

        assert service.get_active_session_count() == 0

        service._active_sessions[uuid4()] = AgentStreamingState()
        assert service.get_active_session_count() == 1

        service._active_sessions[uuid4()] = AgentStreamingState()
        assert service.get_active_session_count() == 2

    def test_is_session_active(
        self,
        mock_session: MagicMock,
    ) -> None:
        """Test checking if session is active."""
        service = AgentStreamingService(mock_session)

        correlation_id = uuid4()
        assert service.is_session_active(correlation_id) is False

        service._active_sessions[correlation_id] = AgentStreamingState()
        assert service.is_session_active(correlation_id) is True

    @pytest.mark.asyncio
    async def test_event_persistence_before_publish(
        self,
        mock_session: MagicMock,
        mock_livekit_service: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test that events are persisted before being published."""

        async def mock_agent_events() -> AsyncGenerator[AgentEvent, None]:
            yield AgentEvent(
                event_type=EventType.UI_BLOCK,
                payload={"type": "inbox_list", "children": []},
            )

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session, mock_livekit_service)
            await service.stream_agent_output(streaming_context, mock_agent_events())

            # All events should be persisted
            assert mock_repo.append_event_raw.call_count == 3  # open + block + close

    @pytest.mark.asyncio
    async def test_session_cleanup_on_completion(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test that session is cleaned up after completion."""

        async def mock_agent_events() -> AsyncGenerator[AgentEvent, None]:
            yield AgentEvent(
                event_type=EventType.ASSISTANT_TEXT_FINAL,
                payload={"text": "Done"},
            )

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)
            await service.stream_agent_output(streaming_context, mock_agent_events())

            # Session should be cleaned up
            assert streaming_context.correlation_id not in service._active_sessions

    @pytest.mark.asyncio
    async def test_session_cleanup_on_error(
        self,
        mock_session: MagicMock,
        streaming_context: StreamingContext,
    ) -> None:
        """Test that session is cleaned up after error."""

        async def mock_agent_events() -> AsyncGenerator[AgentEvent, None]:
            raise RuntimeError("Unexpected error")
            yield  # Make it a generator

        with patch("priority_lens.services.agent_streaming.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = AgentStreamingService(mock_session)

            with pytest.raises(RuntimeError):
                await service.stream_agent_output(streaming_context, mock_agent_events())

            # Session should be cleaned up even after error
            assert streaming_context.correlation_id not in service._active_sessions
