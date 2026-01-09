"""Tests for WatchSubscription model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from rl_emails.models.watch_subscription import WatchSubscription


class TestWatchSubscriptionModel:
    """Tests for WatchSubscription model."""

    def test_create_watch_subscription(self) -> None:
        """Test creating a watch subscription with explicit values."""
        user_id = uuid4()
        subscription = WatchSubscription(
            user_id=user_id,
            status="inactive",
            notification_count=0,
        )

        assert subscription.user_id == user_id
        assert subscription.status == "inactive"
        assert subscription.history_id is None
        assert subscription.expiration is None
        assert subscription.notification_count == 0

    def test_is_active_when_inactive(self) -> None:
        """Test is_active returns False when status is inactive."""
        subscription = WatchSubscription(user_id=uuid4(), status="inactive")

        assert subscription.is_active is False

    def test_is_active_when_active(self) -> None:
        """Test is_active returns True when status is active."""
        subscription = WatchSubscription(user_id=uuid4(), status="active")

        assert subscription.is_active is True

    def test_is_active_when_error(self) -> None:
        """Test is_active returns False when status is error."""
        subscription = WatchSubscription(user_id=uuid4(), status="error")

        assert subscription.is_active is False

    def test_is_expired_when_no_expiration(self) -> None:
        """Test is_expired returns True when no expiration set."""
        subscription = WatchSubscription(user_id=uuid4(), expiration=None)

        assert subscription.is_expired is True

    def test_is_expired_when_expired(self) -> None:
        """Test is_expired returns True when expiration is in the past."""
        past = datetime.now(UTC) - timedelta(hours=1)
        subscription = WatchSubscription(user_id=uuid4(), expiration=past)

        assert subscription.is_expired is True

    def test_is_expired_when_not_expired(self) -> None:
        """Test is_expired returns False when expiration is in the future."""
        future = datetime.now(UTC) + timedelta(days=7)
        subscription = WatchSubscription(user_id=uuid4(), expiration=future)

        assert subscription.is_expired is False

    def test_needs_renewal_when_no_expiration(self) -> None:
        """Test needs_renewal returns True when no expiration set."""
        subscription = WatchSubscription(user_id=uuid4(), expiration=None)

        assert subscription.needs_renewal is True

    def test_needs_renewal_within_24_hours(self) -> None:
        """Test needs_renewal returns True when expiration within 24 hours."""
        soon = datetime.now(UTC) + timedelta(hours=12)
        subscription = WatchSubscription(user_id=uuid4(), expiration=soon)

        assert subscription.needs_renewal is True

    def test_needs_renewal_more_than_24_hours(self) -> None:
        """Test needs_renewal returns False when expiration beyond 24 hours."""
        future = datetime.now(UTC) + timedelta(days=5)
        subscription = WatchSubscription(user_id=uuid4(), expiration=future)

        assert subscription.needs_renewal is False

    def test_repr(self) -> None:
        """Test string representation."""
        subscription = WatchSubscription(user_id=uuid4(), status="active")

        result = repr(subscription)

        assert "WatchSubscription" in result
        assert "active" in result
