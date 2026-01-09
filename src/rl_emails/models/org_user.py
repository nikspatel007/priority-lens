"""Organization user model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_emails.models.base import Base

if TYPE_CHECKING:
    from rl_emails.models.oauth_token import OAuthToken
    from rl_emails.models.organization import Organization
    from rl_emails.models.sync_state import SyncState
    from rl_emails.models.watch_subscription import WatchSubscription


class OrgUser(Base):
    """Organization user model."""

    __tablename__ = "org_users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String, default="member", server_default="member")
    gmail_connected: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
    )
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
    organization: Mapped[Organization] = relationship(
        "Organization",
        back_populates="users",
    )
    oauth_tokens: Mapped[list[OAuthToken]] = relationship(
        "OAuthToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    sync_state: Mapped[SyncState | None] = relationship(
        "SyncState",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    watch_subscription: Mapped[WatchSubscription | None] = relationship(
        "WatchSubscription",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OrgUser(id={self.id}, email={self.email!r}, role={self.role!r})"
