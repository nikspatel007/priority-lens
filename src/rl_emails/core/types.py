"""Shared type definitions."""
from __future__ import annotations

from typing import TypedDict


class EmailData(TypedDict, total=False):
    """Email data structure from parsing."""

    message_id: str
    from_email: str
    from_name: str | None
    to_emails: list[str]
    cc_emails: list[str]
    bcc_emails: list[str]
    subject: str | None
    date_str: str | None
    body_text: str
    body_html: str | None
    headers: dict[str, str]
    labels: list[str]
    in_reply_to: str | None
    references: list[str]


class EmailFeatures(TypedDict, total=False):
    """ML features for an email."""

    email_id: int
    relationship_strength: float
    urgency_score: float
    is_service_email: bool
    service_type: str | None
    service_importance: float


class PriorityScores(TypedDict):
    """Priority scoring components."""

    feature_score: float
    replied_similarity: float
    cluster_novelty: float
    sender_novelty: float
    priority_score: float


class LLMFlags(TypedDict):
    """LLM analysis flags."""

    needs_llm: bool
    reason: str | None


class ClusterAssignment(TypedDict, total=False):
    """Cluster assignments across dimensions."""

    email_id: int
    people_cluster_id: int | None
    content_cluster_id: int | None
    behavior_cluster_id: int | None
    service_cluster_id: int | None
    temporal_cluster_id: int | None
