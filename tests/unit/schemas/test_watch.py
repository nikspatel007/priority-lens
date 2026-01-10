"""Tests for watch subscription schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from priority_lens.schemas.watch import (
    WatchSubscriptionCreate,
    WatchSubscriptionResponse,
    WatchSubscriptionStatus,
    WatchSubscriptionUpdate,
)


class TestWatchSubscriptionCreate:
    """Tests for WatchSubscriptionCreate schema."""

    def test_create_basic(self) -> None:
        """Test creating with required fields."""
        user_id = uuid4()
        schema = WatchSubscriptionCreate(
            user_id=user_id,
            topic_name="projects/test/topics/gmail",
        )

        assert schema.user_id == user_id
        assert schema.topic_name == "projects/test/topics/gmail"
        assert schema.label_ids is None

    def test_create_with_labels(self) -> None:
        """Test creating with label IDs."""
        schema = WatchSubscriptionCreate(
            user_id=uuid4(),
            topic_name="projects/test/topics/gmail",
            label_ids=["INBOX", "IMPORTANT"],
        )

        assert schema.label_ids == ["INBOX", "IMPORTANT"]


class TestWatchSubscriptionUpdate:
    """Tests for WatchSubscriptionUpdate schema."""

    def test_update_empty(self) -> None:
        """Test creating empty update."""
        schema = WatchSubscriptionUpdate()

        assert schema.history_id is None
        assert schema.expiration is None
        assert schema.status is None

    def test_update_with_values(self) -> None:
        """Test creating update with values."""
        now = datetime.now(UTC)
        schema = WatchSubscriptionUpdate(
            history_id="12345",
            expiration=now,
            status="active",
            error_message=None,
        )

        assert schema.history_id == "12345"
        assert schema.expiration == now
        assert schema.status == "active"

    def test_update_status_literal(self) -> None:
        """Test status accepts valid literal values."""
        schema_active = WatchSubscriptionUpdate(status="active")
        schema_inactive = WatchSubscriptionUpdate(status="inactive")
        schema_error = WatchSubscriptionUpdate(status="error")

        assert schema_active.status == "active"
        assert schema_inactive.status == "inactive"
        assert schema_error.status == "error"


class TestWatchSubscriptionResponse:
    """Tests for WatchSubscriptionResponse schema."""

    def test_response_from_model(self) -> None:
        """Test creating response from model-like attributes."""
        now = datetime.now(UTC)
        sub_id = uuid4()
        user_id = uuid4()

        response = WatchSubscriptionResponse(
            id=sub_id,
            user_id=user_id,
            history_id="12345",
            expiration=now,
            topic_name="projects/test/topics/gmail",
            label_ids=["INBOX"],
            status="active",
            last_notification_at=now,
            notification_count=10,
            error_message=None,
            created_at=now,
            updated_at=now,
        )

        assert response.id == sub_id
        assert response.user_id == user_id
        assert response.history_id == "12345"
        assert response.notification_count == 10


class TestWatchSubscriptionStatus:
    """Tests for WatchSubscriptionStatus schema."""

    def test_status_active(self) -> None:
        """Test status for active subscription."""
        now = datetime.now(UTC)
        status = WatchSubscriptionStatus(
            is_active=True,
            is_expired=False,
            needs_renewal=False,
            expiration=now,
            notification_count=5,
            last_notification_at=now,
            error_message=None,
        )

        assert status.is_active is True
        assert status.is_expired is False
        assert status.notification_count == 5

    def test_status_inactive(self) -> None:
        """Test status for inactive subscription."""
        status = WatchSubscriptionStatus(
            is_active=False,
            is_expired=True,
            needs_renewal=True,
            expiration=None,
            notification_count=0,
            last_notification_at=None,
            error_message=None,
        )

        assert status.is_active is False
        assert status.is_expired is True
        assert status.needs_renewal is True

    def test_status_with_error(self) -> None:
        """Test status with error message."""
        status = WatchSubscriptionStatus(
            is_active=False,
            is_expired=False,
            needs_renewal=True,
            expiration=None,
            notification_count=0,
            last_notification_at=None,
            error_message="Watch setup failed",
        )

        assert status.error_message == "Watch setup failed"
