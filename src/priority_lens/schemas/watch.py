"""Watch subscription Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class WatchSubscriptionCreate(BaseModel):
    """Schema for creating a watch subscription."""

    user_id: UUID
    topic_name: str
    label_ids: list[str] | None = None


class WatchSubscriptionUpdate(BaseModel):
    """Schema for updating a watch subscription."""

    history_id: str | None = None
    expiration: datetime | None = None
    topic_name: str | None = None
    label_ids: list[str] | None = None
    status: Literal["inactive", "active", "error"] | None = None
    last_notification_at: datetime | None = None
    error_message: str | None = None


class WatchSubscriptionResponse(BaseModel):
    """Schema for watch subscription response."""

    id: UUID
    user_id: UUID
    history_id: str | None
    expiration: datetime | None
    topic_name: str | None
    label_ids: list[str] | None
    status: str
    last_notification_at: datetime | None
    notification_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WatchSubscriptionStatus(BaseModel):
    """Schema for watch subscription status check."""

    is_active: bool
    is_expired: bool
    needs_renewal: bool
    expiration: datetime | None
    notification_count: int
    last_notification_at: datetime | None
    error_message: str | None
