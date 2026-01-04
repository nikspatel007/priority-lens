"""Feature extraction modules for email RL system."""

from .project import (
    extract_project_features,
    detect_project_mentions,
    detect_deadlines,
    extract_action_items,
)

__all__ = [
    'extract_project_features',
    'detect_project_mentions',
    'detect_deadlines',
    'extract_action_items',
]
