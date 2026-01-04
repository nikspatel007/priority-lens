"""EmailAction dataclass for RL agent action representation.

This module defines the action space for the RL agent when handling emails.
Actions include:
- Primary action type (reply, forward, archive, delete, create_task)
- Priority score (continuous 0-1)
- Response timing recommendations
- Task creation details
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Literal, Optional, TYPE_CHECKING

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

if TYPE_CHECKING:
    import numpy as np


# Action type literals
ActionType = Literal[
    'reply_now',      # Draft immediate reply
    'reply_later',    # Flag for later response
    'forward',        # Forward to someone else
    'archive',        # No action needed
    'delete',         # Spam/irrelevant
    'create_task',    # Convert to task item
]

ResponseTime = Literal[
    'immediate',      # Within 1 hour
    'same_day',       # Within 8 hours
    'next_day',       # Within 24 hours
    'this_week',      # Within 7 days
    'when_possible',  # No urgency
]

TaskPriority = Literal['high', 'medium', 'low']


# Enum versions for tensor indexing
class ActionTypeIndex(IntEnum):
    """Integer indices for action types (for neural network output)."""
    REPLY_NOW = 0
    REPLY_LATER = 1
    FORWARD = 2
    ARCHIVE = 3
    DELETE = 4
    CREATE_TASK = 5


class ResponseTimeIndex(IntEnum):
    """Integer indices for response timing (for neural network output)."""
    IMMEDIATE = 0
    SAME_DAY = 1
    NEXT_DAY = 2
    THIS_WEEK = 3
    WHEN_POSSIBLE = 4


class TaskPriorityIndex(IntEnum):
    """Integer indices for task priority."""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


# Mapping from action type strings to indices
ACTION_TYPE_TO_INDEX: dict[ActionType, int] = {
    'reply_now': ActionTypeIndex.REPLY_NOW,
    'reply_later': ActionTypeIndex.REPLY_LATER,
    'forward': ActionTypeIndex.FORWARD,
    'archive': ActionTypeIndex.ARCHIVE,
    'delete': ActionTypeIndex.DELETE,
    'create_task': ActionTypeIndex.CREATE_TASK,
}

INDEX_TO_ACTION_TYPE: dict[int, ActionType] = {
    v: k for k, v in ACTION_TYPE_TO_INDEX.items()
}

RESPONSE_TIME_TO_INDEX: dict[ResponseTime, int] = {
    'immediate': ResponseTimeIndex.IMMEDIATE,
    'same_day': ResponseTimeIndex.SAME_DAY,
    'next_day': ResponseTimeIndex.NEXT_DAY,
    'this_week': ResponseTimeIndex.THIS_WEEK,
    'when_possible': ResponseTimeIndex.WHEN_POSSIBLE,
}

INDEX_TO_RESPONSE_TIME: dict[int, ResponseTime] = {
    v: k for k, v in RESPONSE_TIME_TO_INDEX.items()
}

TASK_PRIORITY_TO_INDEX: dict[TaskPriority, int] = {
    'high': TaskPriorityIndex.HIGH,
    'medium': TaskPriorityIndex.MEDIUM,
    'low': TaskPriorityIndex.LOW,
}

INDEX_TO_TASK_PRIORITY: dict[int, TaskPriority] = {
    v: k for k, v in TASK_PRIORITY_TO_INDEX.items()
}

# Mapping from training labels (from label_actions.py) to action types
LABEL_TO_ACTION_TYPE: dict[str, ActionType] = {
    'REPLIED': 'reply_now',      # User replied, assume needed urgency
    'FORWARDED': 'forward',
    'DELETED': 'delete',
    'ARCHIVED': 'archive',
    'KEPT': 'reply_later',       # Still in inbox, may need attention
    'COMPOSED': 'reply_now',     # New composition, treat as active
    'JUNK': 'delete',
}


@dataclass
class EmailAction:
    """Agent action output for email handling.

    This represents the complete action recommendation from the RL agent,
    including the primary action type, priority score, and optional
    task creation details.

    Attributes:
        action_type: Primary action to take on the email
        priority: Importance score (0.0=low, 1.0=high)
        suggested_response_time: When to respond (if applicable)
        task_priority: Priority if creating a task
        task_deadline: Deadline if creating a task
        task_assignee: Delegate to someone (if forwarding or delegating task)
        confidence: Model confidence in this action (0-1)
    """
    action_type: ActionType
    priority: float = 0.5
    suggested_response_time: ResponseTime = 'same_day'
    task_priority: TaskPriority = 'medium'
    task_deadline: Optional[datetime] = None
    task_assignee: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate action fields."""
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be between 0 and 1, got {self.priority}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    @property
    def action_type_index(self) -> int:
        """Get integer index for action type (for tensor operations)."""
        return ACTION_TYPE_TO_INDEX[self.action_type]

    @property
    def response_time_index(self) -> int:
        """Get integer index for response time (for tensor operations)."""
        return RESPONSE_TIME_TO_INDEX[self.suggested_response_time]

    @property
    def task_priority_index(self) -> int:
        """Get integer index for task priority."""
        return TASK_PRIORITY_TO_INDEX[self.task_priority]

    @property
    def requires_response(self) -> bool:
        """Check if this action requires user to respond."""
        return self.action_type in ('reply_now', 'reply_later')

    @property
    def requires_immediate_attention(self) -> bool:
        """Check if this action needs immediate attention."""
        return (
            self.action_type == 'reply_now' or
            (self.action_type == 'create_task' and self.task_priority == 'high') or
            self.suggested_response_time == 'immediate'
        )

    @property
    def can_be_automated(self) -> bool:
        """Check if this action can be safely automated."""
        return self.action_type in ('archive', 'delete') and self.confidence > 0.9

    @property
    def should_create_task(self) -> bool:
        """Check if a task should be created."""
        return self.action_type == 'create_task'

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'action_type': self.action_type,
            'priority': self.priority,
            'suggested_response_time': self.suggested_response_time,
            'task_priority': self.task_priority,
            'task_deadline': self.task_deadline.isoformat() if self.task_deadline else None,
            'task_assignee': self.task_assignee,
            'confidence': self.confidence,
        }

    def to_action_vector(self) -> 'np.ndarray':
        """Convert action to a flat numpy array for training.

        Returns:
            Array with shape (9,):
            - [0]: action_type index (0-5)
            - [1]: priority (0-1)
            - [2]: response_time index (0-4)
            - [3]: task_priority index (0-2)
            - [4]: has_deadline (0 or 1)
            - [5]: has_assignee (0 or 1)
            - [6]: confidence (0-1)
            - [7:9]: reserved for future use
        """
        if np is None:
            raise ImportError("numpy is required for to_action_vector()")

        return np.array([
            float(self.action_type_index),
            self.priority,
            float(self.response_time_index),
            float(self.task_priority_index),
            float(self.task_deadline is not None),
            float(self.task_assignee is not None),
            self.confidence,
            0.0,  # Reserved
            0.0,  # Reserved
        ], dtype=np.float32)

    @classmethod
    def from_action_vector(cls, vector: 'np.ndarray') -> 'EmailAction':
        """Create EmailAction from a numpy array.

        Args:
            vector: Array with shape (9,) as produced by to_action_vector()

        Returns:
            EmailAction instance
        """
        action_type_idx = int(round(vector[0]))
        response_time_idx = int(round(vector[2]))
        task_priority_idx = int(round(vector[3]))

        return cls(
            action_type=INDEX_TO_ACTION_TYPE[action_type_idx],
            priority=float(vector[1]),
            suggested_response_time=INDEX_TO_RESPONSE_TIME[response_time_idx],
            task_priority=INDEX_TO_TASK_PRIORITY[task_priority_idx],
            task_deadline=None,  # Not reconstructable from vector
            task_assignee=None,  # Not reconstructable from vector
            confidence=float(vector[6]),
        )

    @classmethod
    def from_training_label(
        cls,
        label: str,
        priority: Optional[float] = None,
        response_hours: Optional[float] = None,
    ) -> 'EmailAction':
        """Create EmailAction from training label.

        Converts labels from label_actions.py (REPLIED, ARCHIVED, etc.)
        to EmailAction instances with inferred attributes.

        Args:
            label: Training label (REPLIED, ARCHIVED, DELETED, FORWARDED, etc.)
            priority: Optional override for priority (0-1)
            response_hours: Hours until user responded (for reply timing inference)

        Returns:
            EmailAction with inferred attributes
        """
        action_type = LABEL_TO_ACTION_TYPE.get(label, 'archive')

        # Infer priority from action type if not provided
        if priority is None:
            priority_defaults = {
                'reply_now': 0.8,
                'reply_later': 0.5,
                'forward': 0.6,
                'archive': 0.3,
                'delete': 0.1,
                'create_task': 0.7,
            }
            priority = priority_defaults.get(action_type, 0.5)

        # Infer response time from hours if available
        response_time: ResponseTime = 'same_day'
        if response_hours is not None:
            if response_hours < 1:
                response_time = 'immediate'
            elif response_hours < 8:
                response_time = 'same_day'
            elif response_hours < 24:
                response_time = 'next_day'
            elif response_hours < 168:  # 7 days
                response_time = 'this_week'
            else:
                response_time = 'when_possible'

        return cls(
            action_type=action_type,
            priority=priority,
            suggested_response_time=response_time,
        )

    @classmethod
    def from_policy_output(
        cls,
        action_logits: 'np.ndarray',
        priority: float,
        timing_logits: 'np.ndarray',
        temperature: float = 1.0,
    ) -> 'EmailAction':
        """Create EmailAction from policy network output.

        Args:
            action_logits: Raw logits for action type (shape: (6,))
            priority: Priority score from network (0-1)
            timing_logits: Raw logits for response timing (shape: (5,))
            temperature: Softmax temperature for sampling

        Returns:
            EmailAction with highest probability action
        """
        if np is None:
            raise ImportError("numpy is required for from_policy_output()")

        # Apply softmax with temperature
        def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
            x = x / temp
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum()

        action_probs = softmax(action_logits, temperature)
        timing_probs = softmax(timing_logits, temperature)

        action_idx = int(np.argmax(action_probs))
        timing_idx = int(np.argmax(timing_probs))
        confidence = float(action_probs[action_idx])

        return cls(
            action_type=INDEX_TO_ACTION_TYPE[action_idx],
            priority=float(np.clip(priority, 0.0, 1.0)),
            suggested_response_time=INDEX_TO_RESPONSE_TIME[timing_idx],
            confidence=confidence,
        )


@dataclass
class ActionProbabilities:
    """Probability distribution over possible actions.

    Used for representing model uncertainty and for exploration
    during training.
    """
    action_probs: dict[ActionType, float] = field(default_factory=dict)
    timing_probs: dict[ResponseTime, float] = field(default_factory=dict)
    priority_mean: float = 0.5
    priority_std: float = 0.1

    def sample_action(self, rng: Optional['np.random.Generator'] = None) -> EmailAction:
        """Sample an action from the probability distribution.

        Args:
            rng: Optional numpy random generator for reproducibility

        Returns:
            Sampled EmailAction
        """
        if np is None:
            raise ImportError("numpy is required for sample_action()")

        if rng is None:
            rng = np.random.default_rng()

        # Sample action type
        action_types = list(self.action_probs.keys())
        probs = np.array([self.action_probs[a] for a in action_types])
        probs = probs / probs.sum()  # Normalize
        action_type = action_types[rng.choice(len(action_types), p=probs)]

        # Sample timing
        timing_types = list(self.timing_probs.keys())
        timing_probs = np.array([self.timing_probs[t] for t in timing_types])
        timing_probs = timing_probs / timing_probs.sum()
        response_time = timing_types[rng.choice(len(timing_types), p=timing_probs)]

        # Sample priority from gaussian
        priority = float(np.clip(
            rng.normal(self.priority_mean, self.priority_std),
            0.0, 1.0
        ))

        return EmailAction(
            action_type=action_type,
            priority=priority,
            suggested_response_time=response_time,
        )

    def entropy(self) -> float:
        """Compute entropy of action distribution (for exploration bonus)."""
        if np is None:
            raise ImportError("numpy is required for entropy()")

        probs = np.array(list(self.action_probs.values()))
        probs = probs[probs > 0]  # Avoid log(0)
        return float(-np.sum(probs * np.log(probs)))


def compute_action_similarity(action1: EmailAction, action2: EmailAction) -> float:
    """Compute similarity score between two actions.

    Used for partial credit in reward computation.

    Args:
        action1: First action
        action2: Second action

    Returns:
        Similarity score (0-1), where 1 means identical
    """
    score = 0.0

    # Exact action type match
    if action1.action_type == action2.action_type:
        score += 0.5
    # Similar action types (partial credit)
    elif is_similar_action(action1.action_type, action2.action_type):
        score += 0.25

    # Priority similarity (inverse of absolute difference)
    priority_sim = 1.0 - abs(action1.priority - action2.priority)
    score += 0.3 * priority_sim

    # Response time similarity
    time_diff = abs(action1.response_time_index - action2.response_time_index)
    time_sim = 1.0 - (time_diff / 4.0)  # 4 is max difference
    score += 0.2 * time_sim

    return score


def is_similar_action(action1: ActionType, action2: ActionType) -> bool:
    """Check if two action types are similar (for partial credit).

    Args:
        action1: First action type
        action2: Second action type

    Returns:
        True if actions are similar enough for partial credit
    """
    # Define similarity groups
    response_actions = {'reply_now', 'reply_later'}
    passive_actions = {'archive', 'delete'}
    task_actions = {'create_task', 'reply_later'}  # Both defer work

    for group in [response_actions, passive_actions, task_actions]:
        if action1 in group and action2 in group:
            return True

    return False


def response_time_to_hours(response_time: ResponseTime) -> float:
    """Convert response time category to expected hours.

    Args:
        response_time: Response time category

    Returns:
        Expected hours until response
    """
    mapping = {
        'immediate': 0.5,
        'same_day': 4.0,
        'next_day': 16.0,
        'this_week': 72.0,
        'when_possible': 168.0,
    }
    return mapping[response_time]


def urgency_from_response_time(hours: float) -> float:
    """Convert response time in hours to urgency score.

    Args:
        hours: Hours until response

    Returns:
        Urgency score (0-1), where 1 is most urgent
    """
    if hours < 1:
        return 1.0
    elif hours < 4:
        return 0.8
    elif hours < 24:
        return 0.6
    elif hours < 72:
        return 0.4
    elif hours < 168:
        return 0.2
    else:
        return 0.1
