"""Project data models with Pydantic validation.

These models represent project-related information extracted from emails:
- Project: Named project or deal
- ProjectMention: A reference to a project in email text
- ProjectFeatures: Computed project-related features
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ProjectMention(BaseModel):
    """A detected project mention in email text.

    Represents a single reference to a project found via pattern
    matching or entity extraction.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, description="The matched project text")
    pattern_type: str = Field(
        ...,
        description="Type of pattern that matched (e.g., 'project_name', 'deal_code')",
    )
    start_pos: int = Field(..., ge=0, description="Start position in source text")
    end_pos: int = Field(..., ge=0, description="End position in source text")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for this match"
    )
    context: Optional[str] = Field(
        default=None, max_length=200, description="Surrounding text context"
    )


class Project(BaseModel):
    """A named project, deal, or initiative.

    Represents a distinct project entity that can be tracked
    across multiple emails.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=200)
    code: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Project code like PROJ-123",
    )
    project_type: Optional[str] = Field(
        default=None,
        description="Type: project, deal, operation, proposal, contract",
    )

    # Status
    is_active: bool = True
    priority: Optional[int] = Field(default=None, ge=0, le=4)

    # Participants
    owner_email: Optional[str] = None
    participants: list[str] = Field(default_factory=list)

    # Timing
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    email_count: int = Field(default=0, ge=0)
    last_activity: Optional[datetime] = None

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProjectFeatures(BaseModel):
    """Computed project-related features for an email.

    These features capture project relevance, deadline presence,
    and action items for ML scoring.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    # Project mentions
    project_mentions: list[ProjectMention] = Field(default_factory=list)
    project_count: int = Field(default=0, ge=0)
    has_project_reference: bool = False

    # Deadline/urgency signals
    has_deadline: bool = False
    deadline_count: int = Field(default=0, ge=0)
    urgency_signal_count: int = Field(default=0, ge=0)
    has_date_mention: bool = False
    date_mention_count: int = Field(default=0, ge=0)

    # Action items
    has_action_request: bool = False
    action_request_count: int = Field(default=0, ge=0)
    has_question: bool = False
    question_count: int = Field(default=0, ge=0)

    # Computed scores (0-1 range)
    project_score: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    action_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Combined project relevance
    overall_project_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Combined score indicating project-related importance",
    )

    def compute_overall_relevance(self) -> float:
        """Compute overall project relevance from component scores."""
        weights = {
            "project": 0.4,
            "urgency": 0.35,
            "action": 0.25,
        }
        return (
            weights["project"] * self.project_score
            + weights["urgency"] * self.urgency_score
            + weights["action"] * self.action_score
        )
