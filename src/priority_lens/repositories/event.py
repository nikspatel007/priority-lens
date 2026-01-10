"""Event repository for database operations."""

from __future__ import annotations

import time
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.models.canonical_event import CanonicalEvent, EventActor, EventType
from priority_lens.schemas.event import EventCreate


class EventRepository:
    """Repository for canonical event database operations.

    This repository ensures append-only semantics for the event log.
    Events cannot be updated or deleted once created.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def append_event(
        self,
        data: EventCreate,
        thread_id: UUID,
        org_id: UUID,
    ) -> CanonicalEvent:
        """Append a new event to the thread.

        This method automatically:
        - Generates a unique event_id
        - Calculates the next seq number (monotonically increasing per thread)
        - Sets the timestamp (ts) to current epoch ms

        Args:
            data: Event creation data.
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            Created canonical event.
        """
        # Get the next sequence number for this thread
        next_seq = await self._get_next_seq(thread_id)

        event = CanonicalEvent(
            event_id=uuid4(),
            thread_id=thread_id,
            org_id=org_id,
            seq=next_seq,
            ts=int(time.time() * 1000),  # Epoch ms
            actor=data.actor.value,
            type=data.type.value,
            payload=data.payload,
            correlation_id=data.correlation_id,
            session_id=data.session_id,
            user_id=data.user_id,
        )
        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)
        return event

    async def append_event_raw(
        self,
        thread_id: UUID,
        org_id: UUID,
        actor: EventActor,
        event_type: EventType,
        payload: dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        session_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> CanonicalEvent:
        """Append a new event using raw parameters.

        Convenience method that doesn't require creating an EventCreate schema.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            actor: Event actor.
            event_type: Event type.
            payload: Event payload.
            correlation_id: Correlation UUID for tracking related events.
            session_id: Session UUID.
            user_id: User UUID.

        Returns:
            Created canonical event.
        """
        data = EventCreate(
            actor=actor,
            type=event_type,
            payload=payload or {},
            correlation_id=correlation_id,
            session_id=session_id,
            user_id=user_id,
        )
        return await self.append_event(data, thread_id, org_id)

    async def get_events_after_seq(
        self,
        thread_id: UUID,
        org_id: UUID,
        after_seq: int = 0,
        limit: int = 100,
    ) -> list[CanonicalEvent]:
        """Get events after a sequence number.

        Used for reconnection to fetch missed events.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            after_seq: Sequence number to fetch events after.
            limit: Maximum number of events to return.

        Returns:
            List of events ordered by seq.
        """
        result = await self.session.execute(
            select(CanonicalEvent)
            .where(
                CanonicalEvent.thread_id == thread_id,
                CanonicalEvent.org_id == org_id,
                CanonicalEvent.seq > after_seq,
            )
            .order_by(CanonicalEvent.seq)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_latest_seq(self, thread_id: UUID) -> int:
        """Get the latest sequence number for a thread.

        Args:
            thread_id: Thread UUID.

        Returns:
            Latest sequence number or 0 if no events exist.
        """
        result = await self.session.execute(
            select(func.max(CanonicalEvent.seq)).where(CanonicalEvent.thread_id == thread_id)
        )
        max_seq = result.scalar_one_or_none()
        return max_seq if max_seq is not None else 0

    async def get_events_by_correlation_id(
        self,
        correlation_id: UUID,
        org_id: UUID,
    ) -> list[CanonicalEvent]:
        """Get all events with a specific correlation ID.

        Args:
            correlation_id: Correlation UUID.
            org_id: Organization UUID.

        Returns:
            List of events ordered by seq.
        """
        result = await self.session.execute(
            select(CanonicalEvent)
            .where(
                CanonicalEvent.correlation_id == correlation_id,
                CanonicalEvent.org_id == org_id,
            )
            .order_by(CanonicalEvent.seq)
        )
        return list(result.scalars().all())

    async def get_events_by_type(
        self,
        thread_id: UUID,
        org_id: UUID,
        event_type: EventType,
        limit: int = 100,
    ) -> list[CanonicalEvent]:
        """Get events of a specific type.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            event_type: Event type to filter by.
            limit: Maximum number of events to return.

        Returns:
            List of events ordered by seq.
        """
        result = await self.session.execute(
            select(CanonicalEvent)
            .where(
                CanonicalEvent.thread_id == thread_id,
                CanonicalEvent.org_id == org_id,
                CanonicalEvent.type == event_type.value,
            )
            .order_by(CanonicalEvent.seq)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_events(self, thread_id: UUID, org_id: UUID) -> int:
        """Count events in a thread.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            Total count of events.
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(CanonicalEvent)
            .where(
                CanonicalEvent.thread_id == thread_id,
                CanonicalEvent.org_id == org_id,
            )
        )
        return result.scalar_one()

    async def _get_next_seq(self, thread_id: UUID) -> int:
        """Get the next sequence number for a thread.

        Args:
            thread_id: Thread UUID.

        Returns:
            Next sequence number.
        """
        latest = await self.get_latest_seq(thread_id)
        return latest + 1
