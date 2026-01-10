"""CanonicalEvent Pydantic schemas."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from priority_lens.models.canonical_event import EventActor, EventType


class EventCreate(BaseModel):
    """Schema for creating a canonical event."""

    actor: EventActor
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: UUID | None = None
    session_id: UUID | None = None
    user_id: UUID | None = None


class EventResponse(BaseModel):
    """Schema for canonical event response."""

    event_id: UUID
    thread_id: UUID
    org_id: UUID
    seq: int
    ts: int
    actor: str
    type: str
    payload: dict[str, Any]
    correlation_id: UUID | None
    session_id: UUID | None
    user_id: UUID | None

    model_config = ConfigDict(from_attributes=True)


class EventListResponse(BaseModel):
    """Schema for list of events."""

    events: list[EventResponse]
    next_seq: int
    has_more: bool
