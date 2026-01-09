"""Gmail watch subscription model for push notifications."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_emails.models.base import Base

if TYPE_CHECKING:
    from rl_emails.models.org_user import OrgUser


class WatchSubscription(Base):
    """Gmail watch subscription for push notifications.

    Tracks Gmail API watch registrations that enable real-time
    email notifications via Google Cloud Pub/Sub.
    """

    __tablename__ = "watch_subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("org_users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Gmail watch details
    history_id: Mapped[str | None] = mapped_column(String, nullable=True)
    expiration: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    topic_name: Mapped[str | None] = mapped_column(String, nullable=True)
    label_ids: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
    )

    # Status tracking
    status: Mapped[str] = mapped_column(
        String,
        default="inactive",
        server_default="inactive",
    )
    last_notification_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    notification_count: Mapped[int] = mapped_column(
        default=0,
        server_default="0",
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    user: Mapped[OrgUser] = relationship(
        "OrgUser",
        back_populates="watch_subscription",
    )

    @property
    def is_active(self) -> bool:
        """Check if the watch subscription is active."""
        return self.status == "active"

    @property
    def is_expired(self) -> bool:
        """Check if the watch subscription has expired."""
        if self.expiration is None:
            return True

        return datetime.now(UTC) >= self.expiration

    @property
    def needs_renewal(self) -> bool:
        """Check if the watch needs renewal (expires within 24 hours)."""
        if self.expiration is None:
            return True
        from datetime import timedelta

        return datetime.now(UTC) >= self.expiration - timedelta(hours=24)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"WatchSubscription(id={self.id}, status={self.status!r})"
