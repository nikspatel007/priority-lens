"""Core utilities for rl-emails."""
from __future__ import annotations

from rl_emails.core.config import Config
from rl_emails.core.db import (
    fetch_count,
    fetch_one_value,
    get_connection,
    get_cursor,
    get_database_url,
)
from rl_emails.core.types import (
    ClusterAssignment,
    EmailData,
    EmailFeatures,
    LLMFlags,
    PriorityScores,
)

__all__ = [
    "Config",
    "ClusterAssignment",
    "EmailData",
    "EmailFeatures",
    "LLMFlags",
    "PriorityScores",
    "fetch_count",
    "fetch_one_value",
    "get_connection",
    "get_cursor",
    "get_database_url",
]
