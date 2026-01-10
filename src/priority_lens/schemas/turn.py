"""Pydantic schemas for conversation turns."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class TurnInputType(str, Enum):
    """Type of turn input."""

    TEXT = "text"
    VOICE = "voice"


class TextInput(BaseModel):
    """Text input for a turn."""

    type: Literal["text"] = "text"
    text: str = Field(..., min_length=1, max_length=10000, description="Text content")


class VoiceInput(BaseModel):
    """Voice transcript input for a turn."""

    type: Literal["voice"] = "voice"
    transcript: str = Field(..., min_length=1, max_length=10000, description="Voice transcript")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Transcription confidence score"
    )
    duration_ms: int | None = Field(
        default=None, ge=0, description="Audio duration in milliseconds"
    )


class TurnCreate(BaseModel):
    """Request to create a new turn."""

    session_id: UUID = Field(..., description="Session ID for this turn")
    input: TextInput | VoiceInput = Field(..., description="Turn input (text or voice)")


class TurnResponse(BaseModel):
    """Response after submitting a turn."""

    correlation_id: UUID = Field(..., description="Correlation ID for tracking this turn")
    accepted: bool = Field(default=True, description="Whether the turn was accepted")
    thread_id: UUID = Field(..., description="Thread ID the turn belongs to")
    session_id: UUID = Field(..., description="Session ID the turn belongs to")
    seq: int = Field(..., ge=0, description="Sequence number of the first event in this turn")


class TurnEventPayload(BaseModel):
    """Base payload for turn events."""

    correlation_id: str = Field(..., description="Turn correlation ID")


class TurnOpenPayload(TurnEventPayload):
    """Payload for turn.user.open event."""

    input_type: str = Field(..., description="Type of input (text or voice)")


class TextSubmitPayload(TurnEventPayload):
    """Payload for ui.text.submit event."""

    text: str = Field(..., description="Submitted text")


class VoiceTranscriptPayload(TurnEventPayload):
    """Payload for stt.final event."""

    transcript: str = Field(..., description="Voice transcript")
    confidence: float = Field(..., description="Transcription confidence")
    duration_ms: int | None = Field(default=None, description="Audio duration")


class TurnClosePayload(TurnEventPayload):
    """Payload for turn.user.close event."""

    reason: str = Field(default="submit", description="Reason for closing the turn")
