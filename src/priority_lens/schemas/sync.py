"""Sync state Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SyncStateUpdate(BaseModel):
    """Schema for updating sync state."""

    sync_status: Literal["idle", "syncing", "error"] | None = None
    last_history_id: str | None = None
    error_message: str | None = None
    emails_synced: int | None = None


class SyncStateResponse(BaseModel):
    """Schema for sync state response."""

    id: UUID
    user_id: UUID
    last_history_id: str | None
    last_sync_at: datetime | None
    sync_status: str
    error_message: str | None
    emails_synced: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# API Response for mobile app sync status polling
SyncStatus = Literal["pending", "syncing", "completed", "failed"]


class SyncStatusApiResponse(BaseModel):
    """API response schema for sync status endpoint.

    Used by mobile app to poll sync progress.
    """

    status: SyncStatus
    emails_synced: int
    total_emails: int | None = None
    progress: float  # 0.0 to 1.0
    error: str | None = None


class CompleteConnectionResponse(BaseModel):
    """API response for completing Google connection.

    Indicates whether a new sync was started.
    """

    status: str
    sync_started: bool
