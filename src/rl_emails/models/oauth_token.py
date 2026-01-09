"""OAuth token model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_emails.models.base import Base

if TYPE_CHECKING:
    from rl_emails.models.org_user import OrgUser


class OAuthToken(Base):
    """OAuth token model for Gmail API authentication."""

    __tablename__ = "oauth_tokens"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("org_users.id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String,
        default="google",
        server_default="google",
    )
    access_token: Mapped[str] = mapped_column(String, nullable=False)
    refresh_token: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    scopes: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
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
        back_populates="oauth_tokens",
    )

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        return datetime.now(self.expires_at.tzinfo) >= self.expires_at

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OAuthToken(id={self.id}, provider={self.provider!r}, expires_at={self.expires_at})"
