"""ConversationThread Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ThreadCreate(BaseModel):
    """Schema for creating a conversation thread."""

    title: str | None = Field(None, max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadUpdate(BaseModel):
    """Schema for updating a conversation thread."""

    title: str | None = Field(None, max_length=500)
    metadata: dict[str, Any] | None = None


class ThreadResponse(BaseModel):
    """Schema for conversation thread response."""

    id: UUID
    org_id: UUID
    user_id: UUID
    title: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_orm_with_metadata(cls, obj: Any) -> ThreadResponse:
        """Create from ORM object handling metadata_ field."""
        return cls(
            id=obj.id,
            org_id=obj.org_id,
            user_id=obj.user_id,
            title=obj.title,
            metadata=obj.metadata_,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
        )


class ThreadListResponse(BaseModel):
    """Schema for list of conversation threads."""

    threads: list[ThreadResponse]
    total: int
