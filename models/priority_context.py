"""Priority context models with Pydantic validation.

These models represent contextual information used for priority scoring:
- SenderContext: Information about the email sender
- ThreadContext: Information about the email thread
- TemporalContext: Time-based features
- UserContext: Information about the email recipient
- PriorityContext: Combined context for priority decisions
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class SenderContext(BaseModel):
    """Context about the email sender for priority scoring.

    Captures relationship strength, communication patterns,
    and organizational hierarchy signals.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    email: str = Field(..., min_length=1)
    name: Optional[str] = None
    domain: Optional[str] = None

    # Communication frequency
    frequency: float = Field(
        default=0.0, ge=0.0, description="Emails per day from this sender"
    )
    emails_7d: int = Field(default=0, ge=0)
    emails_30d: int = Field(default=0, ge=0)
    emails_90d: int = Field(default=0, ge=0)

    # Relationship metrics
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Sender importance score"
    )
    reply_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Rate at which user replies to sender"
    )
    relationship_strength: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall relationship strength"
    )

    # Hierarchy
    org_level: int = Field(
        default=0,
        ge=0,
        le=3,
        description="0=external, 1=peer, 2=manager, 3=executive",
    )
    inferred_hierarchy: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Inferred hierarchy position"
    )

    # Recency
    last_interaction_days: int = Field(
        default=0, ge=0, description="Days since last email from sender"
    )
    first_contact: Optional[datetime] = None
    last_contact: Optional[datetime] = None

    # Classification
    is_important_sender: bool = False
    is_service_sender: bool = False


class ThreadContext(BaseModel):
    """Context about the email thread/conversation.

    Captures thread structure, participation patterns,
    and conversation state.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    thread_id: Optional[str] = None

    # Thread structure
    is_reply: bool = False
    thread_length: int = Field(default=1, ge=1, description="Emails in thread")
    thread_depth: int = Field(default=0, ge=0, description="Reply depth")

    # Participants
    thread_participants: int = Field(default=1, ge=1)
    unique_senders: int = Field(default=1, ge=1)

    # User involvement
    user_already_replied: bool = False
    user_reply_count: int = Field(default=0, ge=0)
    user_first_reply_at: Optional[datetime] = None

    # Timing
    thread_age_hours: float = Field(default=0.0, ge=0.0)
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None

    # Response patterns
    avg_response_time_seconds: Optional[int] = Field(default=None, ge=0)
    thread_duration_seconds: Optional[int] = Field(default=None, ge=0)


class TemporalContext(BaseModel):
    """Time-based features for priority scoring.

    Captures when the email was sent and timing patterns
    that may affect priority.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")

    # Business hours
    is_business_hours: bool = True
    is_weekend: bool = False

    # Recency
    age_hours: float = Field(default=0.0, ge=0.0, description="Hours since sent")
    time_since_last_email: float = Field(
        default=0.0, ge=0.0, description="Hours since any email"
    )

    # Patterns
    is_morning: bool = False  # 6am-12pm
    is_afternoon: bool = False  # 12pm-6pm
    is_evening: bool = False  # 6pm-10pm
    is_night: bool = False  # 10pm-6am

    @classmethod
    def from_datetime(cls, dt: datetime, now: Optional[datetime] = None) -> "TemporalContext":
        """Create TemporalContext from a datetime."""
        if now is None:
            now = datetime.now()

        hour = dt.hour
        dow = dt.weekday()

        return cls(
            timestamp=dt,
            hour_of_day=hour,
            day_of_week=dow,
            is_business_hours=9 <= hour < 18 and dow < 5,
            is_weekend=dow >= 5,
            age_hours=(now - dt).total_seconds() / 3600 if now > dt else 0.0,
            is_morning=6 <= hour < 12,
            is_afternoon=12 <= hour < 18,
            is_evening=18 <= hour < 22,
            is_night=hour >= 22 or hour < 6,
        )


class UserContext(BaseModel):
    """Context about the email recipient (user).

    Captures user profile, workload, and communication patterns
    for personalized priority scoring.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    user_email: str = Field(..., min_length=1)
    user_name: Optional[str] = None
    user_department: str = ""
    user_role: str = ""
    user_manager: Optional[str] = None

    # Communication patterns
    frequent_contacts: dict[str, float] = Field(
        default_factory=dict, description="email -> importance score"
    )
    typical_daily_volume: int = Field(default=50, ge=0)
    avg_response_time_hours: float = Field(default=24.0, ge=0.0)

    # Current state
    current_inbox_size: int = Field(default=0, ge=0)
    unread_count: int = Field(default=0, ge=0)

    # Preferences
    priority_senders: list[str] = Field(default_factory=list)
    priority_domains: list[str] = Field(default_factory=list)
    priority_keywords: list[str] = Field(default_factory=list)


class PriorityContext(BaseModel):
    """Combined context for email priority scoring.

    Aggregates all contextual information needed to compute
    an email's priority score.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # Component contexts
    sender: SenderContext
    thread: ThreadContext
    temporal: TemporalContext
    user: UserContext

    # Pre-computed scores (0-1 range)
    people_score: float = Field(default=0.0, ge=0.0, le=1.0)
    project_score: float = Field(default=0.0, ge=0.0, le=1.0)
    topic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relationship_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Overall priority
    overall_priority: float = Field(default=0.0, ge=0.0, le=1.0)

    def compute_priority(
        self,
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """Compute overall priority from component scores.

        Args:
            weights: Optional weight dict. Defaults to balanced weights.

        Returns:
            Combined priority score (0-1).
        """
        if weights is None:
            weights = {
                "people": 0.25,
                "project": 0.20,
                "topic": 0.15,
                "task": 0.20,
                "temporal": 0.10,
                "relationship": 0.10,
            }

        priority = (
            weights.get("people", 0.0) * self.people_score
            + weights.get("project", 0.0) * self.project_score
            + weights.get("topic", 0.0) * self.topic_score
            + weights.get("task", 0.0) * self.task_score
            + weights.get("temporal", 0.0) * self.temporal_score
            + weights.get("relationship", 0.0) * self.relationship_score
        )

        return min(max(priority, 0.0), 1.0)
