"""Task data models with Pydantic validation.

These models represent task-related information extracted from emails:
- Task: An actionable task extracted from email content
- TaskFeatures: Computed task-related features
- Deadline: A parsed deadline with type and confidence
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class DeadlineType(str, Enum):
    """Types of detected deadlines."""

    EXPLICIT_DATE = "explicit_date"  # "by January 15th"
    EXPLICIT_TIME = "explicit_time"  # "by 3pm today"
    RELATIVE_DAY = "relative_day"  # "by Friday"
    RELATIVE_PERIOD = "relative_period"  # "by end of week"
    URGENCY_KEYWORD = "urgency_keyword"  # "ASAP", "urgent"
    IMPLICIT = "implicit"  # Inferred from context


class TaskComplexity(str, Enum):
    """Estimated effort level for a task."""

    TRIVIAL = "trivial"  # < 5 minutes
    QUICK = "quick"  # 5-30 minutes
    MEDIUM = "medium"  # 30 min - 2 hours
    SUBSTANTIAL = "substantial"  # > 2 hours
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    """Categories of tasks."""

    REVIEW = "review"  # Review document/code
    SEND = "send"  # Send information/document
    SCHEDULE = "schedule"  # Schedule meeting/call
    DECISION = "decision"  # Make a decision
    RESEARCH = "research"  # Research/investigate
    CREATE = "create"  # Create document/artifact
    FOLLOW_UP = "follow_up"  # Follow up on something
    OTHER = "other"


class Deadline(BaseModel):
    """A parsed deadline from email text.

    Represents a single deadline mention with its parsed date,
    source text, and confidence level.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    parsed_date: Optional[datetime] = Field(
        default=None, description="Parsed datetime if successfully extracted"
    )
    source_text: str = Field(..., min_length=1, description="Original text that matched")
    deadline_type: DeadlineType = Field(
        default=DeadlineType.IMPLICIT, description="Type of deadline pattern"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in the parse"
    )
    urgency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Urgency score based on proximity and keywords",
    )
    is_business_hours: bool = Field(
        default=True, description="Whether deadline implies business hours"
    )


class Task(BaseModel):
    """An actionable task extracted from email content.

    Represents a specific task that the email recipient may need
    to complete, with deadline, assignee, and effort information.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    task_id: Optional[str] = Field(default=None, description="Unique task identifier")
    email_id: Optional[int] = Field(default=None, description="Source email ID")

    # Task details
    description: str = Field(..., min_length=1, max_length=1000)
    task_type: TaskType = Field(default=TaskType.OTHER)
    complexity: TaskComplexity = Field(default=TaskComplexity.UNKNOWN)

    # Deadline
    deadline: Optional[datetime] = None
    deadline_text: Optional[str] = None
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Assignment
    assignee_hint: Optional[str] = Field(
        default=None, description="Extracted assignee hint from text"
    )
    is_assigned_to_user: bool = False
    assigned_by: Optional[str] = None

    # Source
    source_text: Optional[str] = Field(
        default=None, max_length=500, description="Original text snippet"
    )

    # Status
    is_completed: bool = False
    completed_at: Optional[datetime] = None

    # Metadata
    created_at: Optional[datetime] = None


class TaskFeatures(BaseModel):
    """Computed task-related features for an email.

    These features indicate whether an email contains actionable
    tasks and provide details for ML scoring.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # Deadline detection
    has_deadline: bool = False
    deadline_date: Optional[datetime] = None
    deadline_text: str = ""
    deadline_urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    deadline_type: Optional[DeadlineType] = None
    all_deadlines: list[Deadline] = Field(default_factory=list)

    # Deliverable detection
    has_deliverable: bool = False
    deliverable_description: str = ""

    # Assignment detection
    is_assigned_to_user: bool = False
    assigned_by: str = ""
    assignment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Effort estimation
    estimated_effort: TaskComplexity = Field(default=TaskComplexity.MEDIUM)

    # Coordination
    requires_others: bool = False
    is_blocker_for_others: bool = False

    # Extracted items
    action_items: list[str] = Field(default_factory=list)
    task_count: int = Field(default=0, ge=0)
    tasks: list[Task] = Field(default_factory=list)

    # Computed score (0-1 range)
    task_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall task importance score",
    )

    def compute_task_score(self) -> float:
        """Compute overall task score from components."""
        score = 0.0

        # Has deadline is significant
        if self.has_deadline:
            score += 0.3 * (1.0 + self.deadline_urgency) / 2.0

        # Assigned to user is significant
        if self.is_assigned_to_user:
            score += 0.25 * self.assignment_confidence

        # Has deliverable
        if self.has_deliverable:
            score += 0.2

        # Blocking others is urgent
        if self.is_blocker_for_others:
            score += 0.15

        # Action items present
        if self.action_items:
            score += 0.1 * min(len(self.action_items) / 3, 1.0)

        return min(score, 1.0)
