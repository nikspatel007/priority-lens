"""ConversationThread repository for database operations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.models.conversation_thread import ConversationThread
from priority_lens.schemas.conversation_thread import ThreadCreate, ThreadUpdate


class ThreadRepository:
    """Repository for conversation thread database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(
        self,
        data: ThreadCreate,
        org_id: UUID,
        user_id: UUID,
    ) -> ConversationThread:
        """Create a new conversation thread.

        Args:
            data: Thread creation data.
            org_id: Organization UUID.
            user_id: User UUID.

        Returns:
            Created conversation thread.
        """
        thread = ConversationThread(
            org_id=org_id,
            user_id=user_id,
            title=data.title,
            metadata_=data.metadata,
        )
        self.session.add(thread)
        await self.session.commit()
        await self.session.refresh(thread)
        return thread

    async def get_by_id(self, thread_id: UUID) -> ConversationThread | None:
        """Get conversation thread by ID.

        Args:
            thread_id: Thread UUID.

        Returns:
            ConversationThread if found, None otherwise.
        """
        result = await self.session.execute(
            select(ConversationThread).where(ConversationThread.id == thread_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_and_org(
        self,
        thread_id: UUID,
        org_id: UUID,
    ) -> ConversationThread | None:
        """Get conversation thread by ID scoped to organization.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            ConversationThread if found, None otherwise.
        """
        result = await self.session.execute(
            select(ConversationThread).where(
                ConversationThread.id == thread_id,
                ConversationThread.org_id == org_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        org_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationThread]:
        """List conversation threads for a user.

        Args:
            user_id: User UUID.
            org_id: Organization UUID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of conversation threads.
        """
        result = await self.session.execute(
            select(ConversationThread)
            .where(
                ConversationThread.user_id == user_id,
                ConversationThread.org_id == org_id,
            )
            .order_by(ConversationThread.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_by_user(self, user_id: UUID, org_id: UUID) -> int:
        """Count conversation threads for a user.

        Args:
            user_id: User UUID.
            org_id: Organization UUID.

        Returns:
            Total count of threads.
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(ConversationThread)
            .where(
                ConversationThread.user_id == user_id,
                ConversationThread.org_id == org_id,
            )
        )
        return result.scalar_one()

    async def update(
        self,
        thread_id: UUID,
        org_id: UUID,
        data: ThreadUpdate,
    ) -> ConversationThread | None:
        """Update a conversation thread.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.
            data: Update data.

        Returns:
            Updated thread if found, None otherwise.
        """
        thread = await self.get_by_id_and_org(thread_id, org_id)
        if thread is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            if key == "metadata":
                thread.metadata_ = value
            else:
                setattr(thread, key, value)

        await self.session.commit()
        await self.session.refresh(thread)
        return thread

    async def delete(self, thread_id: UUID, org_id: UUID) -> bool:
        """Delete a conversation thread.

        Args:
            thread_id: Thread UUID.
            org_id: Organization UUID.

        Returns:
            True if deleted, False if not found.
        """
        thread = await self.get_by_id_and_org(thread_id, org_id)
        if thread is None:
            return False

        await self.session.delete(thread)
        await self.session.commit()
        return True
