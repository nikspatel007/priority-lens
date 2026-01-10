"""Pydantic schemas for LiveKit token generation."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class LiveKitTokenRequest(BaseModel):
    """Request for a LiveKit access token."""

    thread_id: UUID = Field(..., description="Thread ID for the room")
    session_id: UUID = Field(..., description="Session ID for tracking")
    participant_name: str = Field(
        default="user",
        min_length=1,
        max_length=100,
        description="Display name for the participant",
    )
    ttl_seconds: int = Field(
        default=120,
        ge=30,
        le=3600,
        description="Token time-to-live in seconds",
    )


class LiveKitTokenResponse(BaseModel):
    """Response containing a LiveKit access token."""

    token: str = Field(..., description="JWT access token")
    room_name: str = Field(..., description="LiveKit room name")
    livekit_url: str = Field(..., description="LiveKit server URL")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class LiveKitConfig(BaseModel):
    """LiveKit configuration status."""

    enabled: bool = Field(..., description="Whether LiveKit is configured")
    url: str | None = Field(default=None, description="LiveKit server URL")
