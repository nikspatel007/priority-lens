"""Turn service for handling conversation turns."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

from priority_lens.models.canonical_event import CanonicalEvent, EventActor, EventType
from priority_lens.repositories.event import EventRepository
from priority_lens.schemas.turn import (
    TextInput,
    TextSubmitPayload,
    TurnClosePayload,
    TurnCreate,
    TurnOpenPayload,
    TurnResponse,
    VoiceTranscriptPayload,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

# Type alias for session factory
SessionFactory = Callable[[], AbstractAsyncContextManager["AsyncSession"]]


class TurnService:
    """Service for handling conversation turns."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize turn service.

        Args:
            session: Database session.
        """
        self._session = session
        self._event_repo = EventRepository(session)

    async def submit_turn(
        self,
        thread_id: UUID,
        org_id: UUID,
        user_id: UUID,
        turn_data: TurnCreate,
    ) -> TurnResponse:
        """Submit a new conversation turn.

        This creates a sequence of events:
        1. turn.user.open - Indicates user started a turn
        2. ui.text.submit or stt.final - The actual content
        3. turn.user.close - Indicates user finished the turn

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            user_id: User UUID.
            turn_data: Turn input data.

        Returns:
            TurnResponse with correlation_id and sequence info.
        """
        correlation_id = uuid4()
        session_id = turn_data.session_id
        input_data = turn_data.input
        correlation_id_str = str(correlation_id)

        # Determine input type
        input_type = "text" if isinstance(input_data, TextInput) else "voice"

        await logger.ainfo(
            "turn_starting",
            thread_id=str(thread_id),
            session_id=str(session_id),
            correlation_id=correlation_id_str,
            input_type=input_type,
        )

        # Event 1: turn.user.open
        first_event = await self._append_event(
            thread_id=thread_id,
            org_id=org_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            event_type=EventType.TURN_USER_OPEN,
            payload=TurnOpenPayload(
                correlation_id=correlation_id_str,
                input_type=input_type,
            ).model_dump(),
        )

        # Event 2: Content event (ui.text.submit or stt.final)
        if isinstance(input_data, TextInput):
            await self._append_event(
                thread_id=thread_id,
                org_id=org_id,
                session_id=session_id,
                user_id=user_id,
                correlation_id=correlation_id,
                event_type=EventType.UI_TEXT_SUBMIT,
                payload=TextSubmitPayload(
                    correlation_id=correlation_id_str,
                    text=input_data.text,
                ).model_dump(),
            )
        else:
            # VoiceInput
            await self._append_event(
                thread_id=thread_id,
                org_id=org_id,
                session_id=session_id,
                user_id=user_id,
                correlation_id=correlation_id,
                event_type=EventType.STT_FINAL,
                payload=VoiceTranscriptPayload(
                    correlation_id=correlation_id_str,
                    transcript=input_data.transcript,
                    confidence=input_data.confidence,
                    duration_ms=input_data.duration_ms,
                ).model_dump(),
            )

        # Event 3: turn.user.close
        await self._append_event(
            thread_id=thread_id,
            org_id=org_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            event_type=EventType.TURN_USER_CLOSE,
            payload=TurnClosePayload(
                correlation_id=correlation_id_str,
                reason="submit",
            ).model_dump(),
        )

        await logger.ainfo(
            "turn_submitted",
            thread_id=str(thread_id),
            correlation_id=correlation_id_str,
            first_seq=first_event.seq,
        )

        return TurnResponse(
            correlation_id=correlation_id,
            accepted=True,
            thread_id=thread_id,
            session_id=session_id,
            seq=first_event.seq,
        )

    async def _append_event(
        self,
        thread_id: UUID,
        org_id: UUID,
        session_id: UUID,
        user_id: UUID,
        correlation_id: UUID,
        event_type: EventType,
        payload: dict[str, object],
    ) -> CanonicalEvent:
        """Append an event to the thread's event log.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            session_id: Session UUID.
            user_id: User ID string.
            correlation_id: Correlation UUID for the turn.
            event_type: Type of event.
            payload: Event payload.

        Returns:
            Created CanonicalEvent.
        """
        # Use the raw append method which handles seq and ts internally
        event = await self._event_repo.append_event_raw(
            thread_id=thread_id,
            org_id=org_id,
            actor=EventActor.USER,
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
            session_id=session_id,
            user_id=user_id,
        )

        return event

    async def invoke_agent(
        self,
        thread_id: UUID,
        org_id: UUID,
        user_id: UUID,
        session_id: UUID,
        correlation_id: UUID,
        user_message: str,
        session_factory: SessionFactory,
        livekit_room: str | None = None,
    ) -> int:
        """Invoke the agent to respond to a user message.

        This method:
        1. Creates an AgentContext and AgentRunner
        2. Runs the agent to get streaming events
        3. Passes events to AgentStreamingService for persistence

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            user_id: User UUID.
            session_id: Session UUID.
            correlation_id: Correlation UUID for the turn.
            user_message: The user's input text.
            session_factory: Factory for creating database sessions.
            livekit_room: Optional LiveKit room for publishing.

        Returns:
            Number of events emitted by the agent.
        """
        from priority_lens.agent.context import AgentContext
        from priority_lens.agent.graph import AgentRunner
        from priority_lens.services.agent_streaming import (
            AgentStreamingService,
            StreamingContext,
        )

        await logger.ainfo(
            "agent_invoke_start",
            thread_id=str(thread_id),
            correlation_id=str(correlation_id),
        )

        # Create agent context
        from collections.abc import AsyncGenerator

        async def context_session_factory() -> AsyncGenerator[AsyncSession, None]:
            async with session_factory() as sess:
                yield sess

        ctx = AgentContext(
            user_id=user_id,
            org_id=org_id,
            thread_id=thread_id,
            session_factory=context_session_factory,
        )

        # Create streaming context
        streaming_ctx = StreamingContext(
            thread_id=thread_id,
            org_id=org_id,
            session_id=session_id,
            user_id=user_id,
            correlation_id=correlation_id,
            livekit_room=livekit_room,
        )

        # Run agent and stream events
        runner = AgentRunner(ctx)
        agent_events_gen = await runner.run_streaming(user_message)

        streaming_service = AgentStreamingService(self._session)
        emitted_events = await streaming_service.stream_agent_output(
            streaming_ctx, agent_events_gen
        )

        await logger.ainfo(
            "agent_invoke_complete",
            thread_id=str(thread_id),
            correlation_id=str(correlation_id),
            events_count=len(emitted_events),
        )

        return len(emitted_events)
