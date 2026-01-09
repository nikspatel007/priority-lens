"""Sync state model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_emails.models.base import Base

if TYPE_CHECKING:
    from rl_emails.models.org_user import OrgUser


class SyncState(Base):
    """Sync state model for Gmail sync tracking."""

    __tablename__ = "sync_state"

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
    last_history_id: Mapped[str | None] = mapped_column(String, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    sync_status: Mapped[str] = mapped_column(
        String,
        default="idle",
        server_default="idle",
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    emails_synced: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
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
    user: Mapped[OrgUser] = relationship(
        "OrgUser",
        back_populates="sync_state",
    )

    @property
    def is_syncing(self) -> bool:
        """Check if a sync is in progress."""
        return self.sync_status == "syncing"

    @property
    def has_error(self) -> bool:
        """Check if there was a sync error."""
        return self.sync_status == "error"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SyncState(id={self.id}, status={self.sync_status!r}, emails={self.emails_synced})"
