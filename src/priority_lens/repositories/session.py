"""Session repository for database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.models.session import Session
from priority_lens.schemas.session import SessionCreate, SessionUpdate


class SessionRepository:
    """Repository for session database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(
        self,
        data: SessionCreate,
        thread_id: UUID,
        org_id: UUID,
    ) -> Session:
        """Create a new session.

        Args:
            data: Session creation data.
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            Created session.
        """
        sess = Session(
            thread_id=thread_id,
            org_id=org_id,
            mode=data.mode.value,
            livekit_room=data.livekit_room,
            metadata_=data.metadata,
        )
        self.session.add(sess)
        await self.session.commit()
        await self.session.refresh(sess)
        return sess

    async def get_by_id(self, session_id: UUID) -> Session | None:
        """Get session by ID.

        Args:
            session_id: Session UUID.

        Returns:
            Session if found, None otherwise.
        """
        result = await self.session.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    async def get_by_id_and_org(
        self,
        session_id: UUID,
        org_id: UUID,
    ) -> Session | None:
        """Get session by ID scoped to organization.

        Args:
            session_id: Session UUID.
            org_id: Organization UUID.

        Returns:
            Session if found, None otherwise.
        """
        result = await self.session.execute(
            select(Session).where(
                Session.id == session_id,
                Session.org_id == org_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_thread(
        self,
        thread_id: UUID,
        org_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions for a thread.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of sessions.
        """
        result = await self.session.execute(
            select(Session)
            .where(
                Session.thread_id == thread_id,
                Session.org_id == org_id,
            )
            .order_by(Session.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_by_thread(self, thread_id: UUID, org_id: UUID) -> int:
        """Count sessions for a thread.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            Total count of sessions.
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(Session)
            .where(
                Session.thread_id == thread_id,
                Session.org_id == org_id,
            )
        )
        return result.scalar_one()

    async def get_active_session(
        self,
        thread_id: UUID,
        org_id: UUID,
    ) -> Session | None:
        """Get active session for a thread.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            Active session if found, None otherwise.
        """
        result = await self.session.execute(
            select(Session)
            .where(
                Session.thread_id == thread_id,
                Session.org_id == org_id,
                Session.status == "active",
            )
            .order_by(Session.started_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        session_id: UUID,
        org_id: UUID,
        data: SessionUpdate,
    ) -> Session | None:
        """Update a session.

        Args:
            session_id: Session UUID.
            org_id: Organization UUID.
            data: Update data.

        Returns:
            Updated session if found, None otherwise.
        """
        sess = await self.get_by_id_and_org(session_id, org_id)
        if sess is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if key == "metadata":
                sess.metadata_ = value
            elif key == "status":
                sess.status = value.value if hasattr(value, "value") else value
            else:
                setattr(sess, key, value)

        await self.session.commit()
        await self.session.refresh(sess)
        return sess

    async def end_session(
        self,
        session_id: UUID,
        org_id: UUID,
    ) -> Session | None:
        """End a session.

        Args:
            session_id: Session UUID.
            org_id: Organization UUID.

        Returns:
            Ended session if found, None otherwise.
        """
        sess = await self.get_by_id_and_org(session_id, org_id)
        if sess is None:
            return None

        sess.status = "ended"
        sess.ended_at = datetime.now(UTC)

        await self.session.commit()
        await self.session.refresh(sess)
        return sess
