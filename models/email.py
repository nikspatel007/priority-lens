"""Email data models with Pydantic validation.

These models represent email data at various stages of the pipeline:
- RawEmail: Immutable data from MBOX parsing
- Email: Enriched/derived data with parsed fields
- EmailFeatures: Pre-computed ML features
- Attachment: Attachment metadata
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Attachment(BaseModel):
    """Individual attachment metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    content_disposition: Optional[str] = None
    content_hash: Optional[str] = None
    stored_path: Optional[str] = None


class RawEmail(BaseModel):
    """Raw email data from MBOX parsing (immutable source of truth).

    This represents the raw, unparsed email data exactly as extracted
    from the MBOX file. No normalization or parsing is applied.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    message_id: str = Field(..., min_length=1)

    # Raw headers (exactly as parsed from MBOX)
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references_raw: Optional[str] = None

    date_raw: Optional[str] = None
    from_raw: Optional[str] = None
    to_raw: Optional[str] = None
    cc_raw: Optional[str] = None
    bcc_raw: Optional[str] = None
    subject_raw: Optional[str] = None

    # Raw content
    body_text: Optional[str] = None
    body_html: Optional[str] = None

    # Gmail metadata
    labels_raw: Optional[str] = None

    # MBOX metadata
    mbox_offset: Optional[int] = Field(default=None, ge=0)
    raw_size_bytes: Optional[int] = Field(default=None, ge=0)

    imported_at: Optional[datetime] = None


class Email(BaseModel):
    """Enriched email data with parsed and normalized fields.

    This is the primary email model used throughout the pipeline.
    It contains parsed headers, normalized content, and computed
    enrichment fields.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    raw_email_id: Optional[int] = None
    message_id: str = Field(..., min_length=1)
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None

    # Parsed headers
    date_parsed: Optional[datetime] = None
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    to_emails: list[str] = Field(default_factory=list)
    cc_emails: list[str] = Field(default_factory=list)
    subject: Optional[str] = None

    # Content
    body_text: Optional[str] = None
    body_preview: Optional[str] = Field(default=None, max_length=500)
    word_count: Optional[int] = Field(default=None, ge=0)

    # Parsed Gmail metadata
    labels: list[str] = Field(default_factory=list)

    # Attachment summary
    has_attachments: bool = False
    attachment_count: int = Field(default=0, ge=0)
    attachment_types: list[str] = Field(default_factory=list)
    total_attachment_bytes: int = Field(default=0, ge=0)

    # Ownership
    is_sent: bool = False

    # Computed enrichment (Stage 5)
    action: Optional[str] = None
    timing: Optional[str] = None
    response_time_seconds: Optional[int] = Field(default=None, ge=0)
    priority_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Processing metadata
    enriched_at: Optional[datetime] = None
    enrichment_version: int = Field(default=1, ge=1)
    created_at: Optional[datetime] = None

    @field_validator("from_email", "to_emails", "cc_emails", mode="before")
    @classmethod
    def normalize_email_addresses(cls, v):
        """Normalize email addresses to lowercase."""
        if v is None:
            return v
        if isinstance(v, str):
            return v.lower().strip()
        if isinstance(v, list):
            return [e.lower().strip() for e in v if e]
        return v

    @field_validator("labels", mode="before")
    @classmethod
    def normalize_labels(cls, v):
        """Ensure labels is a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [l.strip() for l in v.split(",") if l.strip()]
        return v


class EmailFeatures(BaseModel):
    """Pre-computed ML features for an email.

    These features are computed once and stored for efficient
    retrieval during training and inference.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    email_id: int
    message_id: Optional[str] = None

    # Relationship features
    sender_response_deviation: Optional[float] = None
    sender_frequency_rank: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    inferred_hierarchy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relationship_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    emails_from_sender_7d: int = Field(default=0, ge=0)
    emails_from_sender_30d: int = Field(default=0, ge=0)
    emails_from_sender_90d: int = Field(default=0, ge=0)
    response_rate_to_sender: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_thread_depth: Optional[float] = Field(default=None, ge=0.0)
    days_since_last_email: Optional[float] = Field(default=None, ge=0.0)
    cc_affinity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Service classification
    is_service_email: bool = False
    service_type: Optional[str] = None
    service_email_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    has_list_unsubscribe_header: bool = False
    has_unsubscribe_url: bool = False
    unsubscribe_phrase_count: int = Field(default=0, ge=0)

    # Task features
    task_count: int = Field(default=0, ge=0)
    has_deadline: bool = False
    deadline_urgency: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_assigned_to_user: bool = False
    estimated_effort: Optional[str] = None
    has_deliverable: bool = False

    # Urgency scoring
    urgency_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    urgency_bucket: Optional[str] = None

    # Computed priority scores (0-1 range)
    project_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    topic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    task_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    people_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temporal_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    service_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relationship_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_priority: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Embeddings (stored as arrays)
    feature_vector: Optional[list[float]] = None
    feature_dim: Optional[int] = Field(default=None, ge=0)
    content_embedding: Optional[list[float]] = None
    content_dim: Optional[int] = Field(default=None, ge=0)
    embedding_model: Optional[str] = None

    # Processing metadata
    computed_at: Optional[datetime] = None
    feature_version: int = Field(default=1, ge=1)
    extracted_at: Optional[datetime] = None
    extraction_version: int = Field(default=1, ge=1)
    created_at: Optional[datetime] = None
