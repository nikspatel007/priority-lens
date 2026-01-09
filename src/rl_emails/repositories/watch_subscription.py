"""Watch subscription repository for database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.watch_subscription import WatchSubscription
from rl_emails.schemas.watch import WatchSubscriptionUpdate


class WatchSubscriptionRepository:
    """Repository for watch subscription database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def get_or_create(self, user_id: UUID) -> WatchSubscription:
        """Get or create watch subscription for a user.

        Args:
            user_id: User UUID.

        Returns:
            Watch subscription (existing or newly created).
        """
        subscription = await self.get_by_user_id(user_id)
        if subscription is None:
            subscription = WatchSubscription(user_id=user_id)
            self.session.add(subscription)
            await self.session.commit()
            await self.session.refresh(subscription)
        return subscription

    async def get_by_user_id(self, user_id: UUID) -> WatchSubscription | None:
        """Get watch subscription by user ID.

        Args:
            user_id: User UUID.

        Returns:
            Watch subscription if found, None otherwise.
        """
        result = await self.session.execute(
            select(WatchSubscription).where(WatchSubscription.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def update(
        self, user_id: UUID, data: WatchSubscriptionUpdate
    ) -> WatchSubscription | None:
        """Update watch subscription.

        Args:
            user_id: User UUID.
            data: Update data.

        Returns:
            Updated watch subscription if found, None otherwise.
        """
        subscription = await self.get_by_user_id(user_id)
        if subscription is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(subscription, key, value)

        await self.session.commit()
        await self.session.refresh(subscription)
        return subscription

    async def activate(
        self,
        user_id: UUID,
        history_id: str,
        expiration: datetime,
        topic_name: str,
        label_ids: list[str] | None = None,
    ) -> WatchSubscription:
        """Activate a watch subscription.

        Args:
            user_id: User UUID.
            history_id: Gmail history ID from watch setup.
            expiration: When the watch expires.
            topic_name: Pub/Sub topic name.
            label_ids: Labels being watched.

        Returns:
            Updated watch subscription.
        """
        subscription = await self.get_or_create(user_id)
        subscription.history_id = history_id
        subscription.expiration = expiration
        subscription.topic_name = topic_name
        subscription.label_ids = label_ids
        subscription.status = "active"
        subscription.error_message = None
        await self.session.commit()
        await self.session.refresh(subscription)
        return subscription

    async def deactivate(self, user_id: UUID) -> WatchSubscription | None:
        """Deactivate a watch subscription.

        Args:
            user_id: User UUID.

        Returns:
            Updated watch subscription if found, None otherwise.
        """
        subscription = await self.get_by_user_id(user_id)
        if subscription is None:
            return None

        subscription.status = "inactive"
        subscription.expiration = None
        await self.session.commit()
        await self.session.refresh(subscription)
        return subscription

    async def record_notification(self, user_id: UUID) -> WatchSubscription | None:
        """Record that a notification was received.

        Args:
            user_id: User UUID.

        Returns:
            Updated watch subscription if found, None otherwise.
        """
        subscription = await self.get_by_user_id(user_id)
        if subscription is None:
            return None

        subscription.last_notification_at = datetime.now(UTC)
        subscription.notification_count += 1
        await self.session.commit()
        await self.session.refresh(subscription)
        return subscription

    async def set_error(self, user_id: UUID, error_message: str) -> WatchSubscription:
        """Set error state on a watch subscription.

        Args:
            user_id: User UUID.
            error_message: Error description.

        Returns:
            Updated watch subscription.
        """
        subscription = await self.get_or_create(user_id)
        subscription.status = "error"
        subscription.error_message = error_message
        await self.session.commit()
        await self.session.refresh(subscription)
        return subscription

    async def get_expiring_soon(self, hours: int = 24) -> list[WatchSubscription]:
        """Get active subscriptions expiring within specified hours.

        Args:
            hours: Number of hours to look ahead.

        Returns:
            List of watch subscriptions that need renewal.
        """
        from datetime import timedelta

        threshold = datetime.now(UTC) + timedelta(hours=hours)
        result = await self.session.execute(
            select(WatchSubscription).where(
                WatchSubscription.status == "active",
                WatchSubscription.expiration <= threshold,
            )
        )
        return list(result.scalars().all())

    async def get_all_active(self) -> list[WatchSubscription]:
        """Get all active watch subscriptions.

        Returns:
            List of active watch subscriptions.
        """
        result = await self.session.execute(
            select(WatchSubscription).where(WatchSubscription.status == "active")
        )
        return list(result.scalars().all())

    async def delete(self, user_id: UUID) -> bool:
        """Delete a watch subscription.

        Args:
            user_id: User UUID.

        Returns:
            True if deleted, False if not found.
        """
        subscription = await self.get_by_user_id(user_id)
        if subscription is None:
            return False

        await self.session.delete(subscription)
        await self.session.commit()
        return True
