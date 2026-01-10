"""Session Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SessionMode(str, Enum):
    """Session mode types."""

    TEXT = "text"
    VOICE = "voice"


class SessionStatus(str, Enum):
    """Session status types."""

    ACTIVE = "active"
    ENDED = "ended"


class SessionCreate(BaseModel):
    """Schema for creating a session."""

    mode: SessionMode = SessionMode.TEXT
    livekit_room: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    """Schema for updating a session."""

    status: SessionStatus | None = None
    livekit_room: str | None = None
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    """Schema for session response."""

    id: UUID
    thread_id: UUID
    org_id: UUID
    mode: str
    status: str
    livekit_room: str | None
    metadata: dict[str, Any]
    started_at: datetime
    ended_at: datetime | None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_orm_with_metadata(cls, obj: Any) -> SessionResponse:
        """Create from ORM object handling metadata_ field."""
        return cls(
            id=obj.id,
            thread_id=obj.thread_id,
            org_id=obj.org_id,
            mode=obj.mode,
            status=obj.status,
            livekit_room=obj.livekit_room,
            metadata=obj.metadata_,
            started_at=obj.started_at,
            ended_at=obj.ended_at,
        )


class SessionListResponse(BaseModel):
    """Schema for list of sessions."""

    sessions: list[SessionResponse]
    total: int
