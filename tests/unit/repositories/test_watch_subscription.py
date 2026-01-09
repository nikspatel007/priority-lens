"""Tests for watch subscription repository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from rl_emails.models.watch_subscription import WatchSubscription
from rl_emails.repositories.watch_subscription import WatchSubscriptionRepository
from rl_emails.schemas.watch import WatchSubscriptionUpdate


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def repository(mock_session: AsyncMock) -> WatchSubscriptionRepository:
    """Create repository with mock session."""
    return WatchSubscriptionRepository(mock_session)


class TestWatchSubscriptionRepository:
    """Tests for WatchSubscriptionRepository."""

    @pytest.mark.asyncio
    async def test_get_by_user_id_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting subscription by user ID when found."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="active",
            notification_count=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_user_id(user_id)

        assert result == subscription

    @pytest.mark.asyncio
    async def test_get_by_user_id_not_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting subscription by user ID when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_user_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_existing(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test get_or_create returns existing subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="active",
            notification_count=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        result = await repository.get_or_create(user_id)

        assert result == subscription
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_new(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test get_or_create creates new subscription."""
        user_id = uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_or_create(user_id)

        assert isinstance(result, WatchSubscription)
        assert result.user_id == user_id
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating existing subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="inactive",
            notification_count=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        update_data = WatchSubscriptionUpdate(status="active", history_id="12345")
        result = await repository.update(user_id, update_data)

        assert result is not None
        assert result.status == "active"
        assert result.history_id == "12345"
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test updating non-existent subscription."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        update_data = WatchSubscriptionUpdate(status="active")
        result = await repository.update(uuid4(), update_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_activate(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test activating a subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="inactive",
            notification_count=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        expiration = datetime.now(UTC) + timedelta(days=7)
        result = await repository.activate(
            user_id=user_id,
            history_id="history123",
            expiration=expiration,
            topic_name="projects/test/topics/gmail",
            label_ids=["INBOX"],
        )

        assert result.status == "active"
        assert result.history_id == "history123"
        assert result.expiration == expiration
        assert result.topic_name == "projects/test/topics/gmail"
        assert result.label_ids == ["INBOX"]
        assert result.error_message is None
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test deactivating existing subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="active",
            notification_count=0,
            expiration=datetime.now(UTC) + timedelta(days=7),
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        result = await repository.deactivate(user_id)

        assert result is not None
        assert result.status == "inactive"
        assert result.expiration is None
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_not_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test deactivating non-existent subscription."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.deactivate(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_record_notification_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test recording notification for existing subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="active",
            notification_count=5,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        result = await repository.record_notification(user_id)

        assert result is not None
        assert result.notification_count == 6
        assert result.last_notification_at is not None
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_notification_not_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test recording notification for non-existent subscription."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.record_notification(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_set_error(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test setting error state on subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="active",
            notification_count=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        result = await repository.set_error(user_id, "Watch expired")

        assert result.status == "error"
        assert result.error_message == "Watch expired"
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_expiring_soon(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting subscriptions expiring soon."""
        subscriptions = [
            WatchSubscription(
                id=uuid4(),
                user_id=uuid4(),
                status="active",
                notification_count=0,
                expiration=datetime.now(UTC) + timedelta(hours=12),
            ),
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = subscriptions
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_expiring_soon(hours=24)

        assert len(result) == 1
        assert result[0].status == "active"

    @pytest.mark.asyncio
    async def test_get_all_active(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test getting all active subscriptions."""
        subscriptions = [
            WatchSubscription(
                id=uuid4(),
                user_id=uuid4(),
                status="active",
                notification_count=0,
            ),
            WatchSubscription(
                id=uuid4(),
                user_id=uuid4(),
                status="active",
                notification_count=5,
            ),
        ]

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = subscriptions
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_all_active()

        assert len(result) == 2
        assert all(s.status == "active" for s in result)

    @pytest.mark.asyncio
    async def test_delete_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting existing subscription."""
        user_id = uuid4()
        subscription = WatchSubscription(
            id=uuid4(),
            user_id=user_id,
            status="active",
            notification_count=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = subscription
        mock_session.execute.return_value = mock_result

        result = await repository.delete(user_id)

        assert result is True
        mock_session.delete.assert_called_once_with(subscription)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self, repository: WatchSubscriptionRepository, mock_session: AsyncMock
    ) -> None:
        """Test deleting non-existent subscription."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.delete(uuid4())

        assert result is False
