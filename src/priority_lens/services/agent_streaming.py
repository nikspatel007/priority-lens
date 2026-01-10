"""Agent streaming service for real-time event publishing."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from priority_lens.models.canonical_event import EventActor, EventType
from priority_lens.repositories.event import EventRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from priority_lens.services.livekit_service import LiveKitService

logger = structlog.get_logger(__name__)


class AgentCancellationError(Exception):
    """Raised when agent execution is cancelled."""

    pass


@dataclass
class StreamingContext:
    """Context for agent streaming operations."""

    thread_id: UUID
    org_id: UUID
    session_id: UUID
    user_id: UUID
    correlation_id: UUID
    livekit_room: str | None = None


@dataclass
class AgentEvent:
    """Event emitted by the agent during streaming."""

    event_type: EventType
    payload: dict[str, Any]
    actor: EventActor = EventActor.AGENT


@dataclass
class AgentStreamingState:
    """Mutable state for an agent streaming session."""

    is_cancelled: bool = False
    events_emitted: int = 0
    turn_open: bool = False


class AgentStreamingService:
    """Service for streaming agent events to LiveKit and database.

    This service handles:
    - Event persistence to the canonical event log
    - Event publishing to LiveKit data channel
    - Turn lifecycle management (open/close)
    - Cancellation/barge-in support
    """

    # Active streaming sessions for cancellation support
    _active_sessions: dict[UUID, AgentStreamingState] = field(default_factory=dict)

    def __init__(
        self,
        session: AsyncSession,
        livekit_service: LiveKitService | None = None,
    ) -> None:
        """Initialize agent streaming service.

        Args:
            session: Database session.
            livekit_service: Optional LiveKit service for publishing.
        """
        self._session = session
        self._event_repo = EventRepository(session)
        self._livekit = livekit_service
        self._active_sessions: dict[UUID, AgentStreamingState] = {}

    async def stream_agent_output(
        self,
        ctx: StreamingContext,
        agent_events: AsyncGenerator[AgentEvent, None],
    ) -> list[AgentEvent]:
        """Stream agent output to database and LiveKit.

        This method:
        1. Emits turn.agent.open event
        2. Streams all agent events, persisting each to DB
        3. Publishes events to LiveKit if configured
        4. Emits turn.agent.close event

        Args:
            ctx: Streaming context with IDs.
            agent_events: Async generator of agent events.

        Returns:
            List of emitted events.

        Raises:
            AgentCancellationError: If streaming was cancelled.
        """
        # Initialize streaming state
        state = AgentStreamingState()
        self._active_sessions[ctx.correlation_id] = state

        emitted_events: list[AgentEvent] = []

        try:
            # Emit turn.agent.open
            open_event = AgentEvent(
                event_type=EventType.TURN_AGENT_OPEN,
                payload={
                    "correlation_id": str(ctx.correlation_id),
                },
            )
            await self._emit_event(ctx, open_event)
            emitted_events.append(open_event)
            state.turn_open = True

            # Stream agent events
            async for event in agent_events:
                # Check for cancellation
                if state.is_cancelled:
                    raise AgentCancellationError("Agent execution cancelled")

                await self._emit_event(ctx, event)
                emitted_events.append(event)
                state.events_emitted += 1

            # Emit turn.agent.close with success
            close_event = AgentEvent(
                event_type=EventType.TURN_AGENT_CLOSE,
                payload={
                    "correlation_id": str(ctx.correlation_id),
                    "reason": "complete",
                    "events_emitted": state.events_emitted,
                },
            )
            await self._emit_event(ctx, close_event)
            emitted_events.append(close_event)

        except AgentCancellationError:
            # Emit turn.agent.close with cancel reason
            if state.turn_open:
                close_event = AgentEvent(
                    event_type=EventType.TURN_AGENT_CLOSE,
                    payload={
                        "correlation_id": str(ctx.correlation_id),
                        "reason": "cancelled",
                        "events_emitted": state.events_emitted,
                    },
                )
                await self._emit_event(ctx, close_event)
                emitted_events.append(close_event)
            raise

        except Exception as e:
            # Emit turn.agent.close with error reason
            if state.turn_open:
                close_event = AgentEvent(
                    event_type=EventType.TURN_AGENT_CLOSE,
                    payload={
                        "correlation_id": str(ctx.correlation_id),
                        "reason": "error",
                        "error": str(e),
                        "events_emitted": state.events_emitted,
                    },
                )
                await self._emit_event(ctx, close_event)
                emitted_events.append(close_event)
            raise

        finally:
            # Clean up active session
            self._active_sessions.pop(ctx.correlation_id, None)

        return emitted_events

    async def cancel_agent(
        self,
        correlation_id: UUID,
    ) -> bool:
        """Cancel an active agent streaming session.

        Args:
            correlation_id: Correlation ID of the session to cancel.

        Returns:
            True if session was found and cancelled, False otherwise.
        """
        state = self._active_sessions.get(correlation_id)
        if state is None:
            await logger.ainfo(
                "cancel_agent_not_found",
                correlation_id=str(correlation_id),
            )
            return False

        state.is_cancelled = True
        await logger.ainfo(
            "agent_cancelled",
            correlation_id=str(correlation_id),
            events_emitted=state.events_emitted,
        )
        return True

    async def emit_system_cancel_event(
        self,
        ctx: StreamingContext,
        reason: str = "user_request",
    ) -> None:
        """Emit a system.cancel event.

        Args:
            ctx: Streaming context.
            reason: Reason for cancellation.
        """
        event = AgentEvent(
            event_type=EventType.SYSTEM_CANCEL,
            payload={
                "correlation_id": str(ctx.correlation_id),
                "reason": reason,
            },
            actor=EventActor.SYSTEM,
        )
        await self._emit_event(ctx, event)

    async def _emit_event(
        self,
        ctx: StreamingContext,
        event: AgentEvent,
    ) -> None:
        """Emit an event to database and LiveKit.

        Events are persisted before publishing to ensure durability.

        Args:
            ctx: Streaming context.
            event: Event to emit.
        """
        # Persist to database first (durability)
        await self._event_repo.append_event_raw(
            thread_id=ctx.thread_id,
            org_id=ctx.org_id,
            actor=event.actor,
            event_type=event.event_type,
            payload=event.payload,
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            user_id=ctx.user_id,
        )

        # Publish to LiveKit if configured and room specified
        if self._livekit is not None and ctx.livekit_room is not None:
            await self._publish_to_livekit(ctx.livekit_room, event)

    async def _publish_to_livekit(
        self,
        room_name: str,
        event: AgentEvent,
    ) -> None:
        """Publish event to LiveKit data channel.

        Args:
            room_name: LiveKit room name.
            event: Event to publish.
        """
        # LiveKit data channel publishing will be implemented
        # when we have the room API client
        # For now, just log the intent
        await logger.adebug(
            "livekit_publish_event",
            room=room_name,
            event_type=event.event_type.value,
        )

    def get_active_session_count(self) -> int:
        """Get the number of active streaming sessions.

        Returns:
            Number of active sessions.
        """
        return len(self._active_sessions)

    def is_session_active(self, correlation_id: UUID) -> bool:
        """Check if a streaming session is active.

        Args:
            correlation_id: Correlation ID to check.

        Returns:
            True if session is active, False otherwise.
        """
        return correlation_id in self._active_sessions
