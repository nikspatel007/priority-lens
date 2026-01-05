"""Pydantic models for rl-emails data validation.

This module provides type-safe, validated data models for the email
prioritization pipeline. All data passing between pipeline stages
should use these models to ensure consistency and catch errors early.
"""

from .email import (
    Email,
    RawEmail,
    EmailFeatures,
    Attachment,
)
from .project import (
    Project,
    ProjectMention,
    ProjectFeatures,
)
from .task import (
    Task,
    TaskFeatures,
    Deadline,
    DeadlineType,
    TaskComplexity,
    TaskType,
)
from .priority_context import (
    PriorityContext,
    SenderContext,
    ThreadContext,
    TemporalContext,
    UserContext,
)

__all__ = [
    # Email models
    "Email",
    "RawEmail",
    "EmailFeatures",
    "Attachment",
    # Project models
    "Project",
    "ProjectMention",
    "ProjectFeatures",
    # Task models
    "Task",
    "TaskFeatures",
    "Deadline",
    "DeadlineType",
    "TaskComplexity",
    "TaskType",
    # Priority context models
    "PriorityContext",
    "SenderContext",
    "ThreadContext",
    "TemporalContext",
    "UserContext",
]
