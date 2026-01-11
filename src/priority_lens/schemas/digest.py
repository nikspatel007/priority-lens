"""Digest schema definitions for the Smart Digest feature."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UrgencyLevel(str, Enum):
    """Urgency level for digest items."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DigestItemType(str, Enum):
    """Type of digest item."""

    TODO = "todo"
    TOPIC = "topic"


class DigestAction(BaseModel):
    """Action available on a digest item."""

    id: str = Field(..., description="Action identifier")
    type: str = Field(..., description="Action type (e.g., 'confirm_meeting', 'reply')")
    label: str = Field(..., description="Display label for the action")
    endpoint: str | None = Field(None, description="API endpoint if needed")
    params: dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class DigestTodoItem(BaseModel):
    """A suggested to-do item in the digest."""

    id: str = Field(..., description="Unique item identifier")
    title: str = Field(..., description="Brief description of the to-do")
    source: str = Field(..., description="Where this to-do originated (e.g., 'Email from Alex')")
    urgency: UrgencyLevel = Field(..., description="Urgency level")
    due: str | None = Field(None, description="Due date/time (e.g., 'Today', 'Tomorrow')")
    context: str | None = Field(None, description="Additional context")
    email_id: str | None = Field(None, description="Related email ID if applicable")
    actions: list[DigestAction] = Field(default_factory=list, description="Available actions")


class DigestTopicItem(BaseModel):
    """A topic to catch up on in the digest."""

    id: str = Field(..., description="Unique topic identifier")
    title: str = Field(..., description="Topic name/summary")
    email_count: int = Field(..., description="Number of related emails")
    participants: list[str] = Field(default_factory=list, description="People involved")
    last_activity: str = Field(..., description="When last activity occurred")
    summary: str | None = Field(None, description="AI-generated summary")
    urgency: UrgencyLevel = Field(UrgencyLevel.LOW, description="Urgency level")


class DigestSection(BaseModel):
    """A section in the digest view."""

    id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    icon: str | None = Field(None, description="Icon name for the section")
    todos: list[DigestTodoItem] = Field(default_factory=list, description="To-do items")
    topics: list[DigestTopicItem] = Field(default_factory=list, description="Topic items")


class DigestResponse(BaseModel):
    """Complete digest response."""

    greeting: str = Field(..., description="Personalized greeting")
    subtitle: str = Field(..., description="Summary subtitle (e.g., '3 items need attention')")
    suggested_todos: list[DigestTodoItem] = Field(
        default_factory=list, description="Actionable to-do items"
    )
    topics_to_catchup: list[DigestTopicItem] = Field(
        default_factory=list, description="Topics to review"
    )
    last_updated: datetime = Field(..., description="When digest was generated")
    user_preferences: dict[str, Any] = Field(
        default_factory=dict, description="User preference data for UI customization"
    )
