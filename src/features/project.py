#!/usr/bin/env python3
"""Project score extraction from email content.

Extracts features related to:
- Project mentions (named projects, project codes)
- Deadlines (dates, urgency markers)
- Action items (tasks, requests, TODOs)
"""

import re
from dataclasses import dataclass
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


# Project name patterns
PROJECT_PATTERNS = [
    r'\b[Pp]roject\s+[A-Z][a-zA-Z]+\b',  # "Project Eagle", "Project Alpha"
    r'\b[A-Z]{2,5}-\d{2,6}\b',  # "PROJ-123", "ENR-45678"
    r'\b[Dd]eal\s+[A-Z][a-zA-Z]+\b',  # "Deal Raptor"
    r'\b[Oo]peration\s+[A-Z][a-zA-Z]+\b',  # "Operation Sunrise"
    r'\bphase\s+[IVX\d]+\b',  # "Phase I", "Phase 2"
    r'\b[Pp]roposal\s+#?\d+\b',  # "Proposal #123"
    r'\b[Cc]ontract\s+#?\d+\b',  # "Contract #456"
]

# Deadline/urgency patterns
DEADLINE_PATTERNS = [
    r'\bASAP\b',
    r'\b[Uu]rgent\b',
    r'\b[Ii]mportant\b',
    r'\bEOD\b',
    r'\bEOB\b',  # End of business
    r'\bCOB\b',  # Close of business
    r'\bby\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:day|week|month)\b',
    r'\bby\s+(?:today|tomorrow|tonight|Monday|Tuesday|Wednesday|Thursday|Friday)\b',
    r'\bdue\s+(?:by|on|today|tomorrow)\b',
    r'\bdeadline\b',
    r'\btime[-\s]?sensitive\b',
    r'\bpriority\b',
    r'\bimmediately\b',
    r'\bright\s+away\b',
    r'\bas\s+soon\s+as\s+possible\b',
]

# Date patterns
DATE_PATTERNS = [
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 1/15/2001
    r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # 1-15-2001
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b',
]

# Action item patterns (imperative/request language)
ACTION_PATTERNS = [
    r'\b[Pp]lease\s+(?:review|send|update|confirm|check|call|respond|reply|forward|prepare|complete)\b',
    r'\b[Nn]eed\s+(?:you\s+)?to\b',
    r'\b[Cc]an\s+you\b',
    r'\b[Cc]ould\s+you\b',
    r'\b[Ww]ould\s+you\b',
    r'\b[Ll]et\s+me\s+know\b',
    r'\b[Gg]et\s+back\s+to\s+me\b',
    r'\b[Ff]ollow\s+up\b',
    r'\b[Aa]ction\s+(?:item|required)\b',
    r'\bTODO\b',
    r'\bFYI\b',
    r'\b[Aa]ttached\s+(?:is|are|please\s+find)\b',
    r'\b[Ss]ee\s+attached\b',
    r'\b[Rr]eview\s+(?:and|the)\b',
    r'\b[Aa]pproval\s+(?:needed|required|requested)\b',
    r'\b[Ss]ign(?:ature)?\s+(?:needed|required)\b',
    r'\b[Aa]waiting\s+(?:your)?\b',
    r'\b[Pp]ending\s+(?:your)?\b',
]

# Question patterns
QUESTION_PATTERNS = [
    r'\?',  # Question mark
    r'\b[Ww]hat\s+(?:is|are|do|does|did|will|would|should|can|could)\b',
    r'\b[Ww]hen\s+(?:is|are|do|does|did|will|would|should|can|could)\b',
    r'\b[Ww]here\s+(?:is|are|do|does|did|will|would|should|can|could)\b',
    r'\b[Ww]ho\s+(?:is|are|will|would|should|can|could)\b',
    r'\b[Ww]hy\s+(?:is|are|do|does|did|will|would|should)\b',
    r'\b[Hh]ow\s+(?:is|are|do|does|did|will|would|should|can|could|much|many|long)\b',
    r'\b[Dd]o\s+you\b',
    r'\b[Aa]re\s+you\b',
    r'\b[Cc]an\s+you\b',
    r'\b[Ww]ill\s+you\b',
]


@dataclass
class ProjectMention:
    """A detected project mention in email text."""
    text: str
    pattern_type: str
    start: int
    end: int


@dataclass
class Deadline:
    """A detected deadline or urgency marker."""
    text: str
    deadline_type: str  # 'date', 'urgency', 'time_bound'
    start: int
    end: int


@dataclass
class ActionItem:
    """A detected action item or request."""
    text: str
    action_type: str  # 'request', 'question', 'todo', 'approval'
    start: int
    end: int


@dataclass
class ProjectFeatures:
    """Extracted project-related features from an email."""
    # Counts
    project_mention_count: int
    deadline_count: int
    action_item_count: int
    question_count: int

    # Scores (0-1)
    project_score: float      # Overall project relevance
    urgency_score: float      # How urgent/time-sensitive
    action_score: float       # How many actions requested

    # Lists of detected items
    project_mentions: list[ProjectMention]
    deadlines: list[Deadline]
    action_items: list[ActionItem]

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numpy array (or list if numpy unavailable) for ML pipeline.

        Returns 8-dimensional vector:
        [project_count, deadline_count, action_count, question_count,
         project_score, urgency_score, action_score, has_any_project_content]
        """
        values = [
            min(self.project_mention_count, 10) / 10.0,  # Normalized count
            min(self.deadline_count, 5) / 5.0,
            min(self.action_item_count, 10) / 10.0,
            min(self.question_count, 10) / 10.0,
            self.project_score,
            self.urgency_score,
            self.action_score,
            1.0 if (self.project_mention_count > 0 or self.deadline_count > 0) else 0.0,
        ]
        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


def detect_project_mentions(text: str) -> list[ProjectMention]:
    """Detect project mentions in email text.

    Args:
        text: Email body or subject text

    Returns:
        List of ProjectMention objects
    """
    mentions = []

    for pattern in PROJECT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            mentions.append(ProjectMention(
                text=match.group(),
                pattern_type='project_name',
                start=match.start(),
                end=match.end(),
            ))

    # Deduplicate overlapping matches
    mentions = _deduplicate_spans(mentions)

    return mentions


def detect_deadlines(text: str) -> list[Deadline]:
    """Detect deadlines and urgency markers in email text.

    Args:
        text: Email body or subject text

    Returns:
        List of Deadline objects
    """
    deadlines = []

    # Check urgency patterns
    for pattern in DEADLINE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            deadlines.append(Deadline(
                text=match.group(),
                deadline_type='urgency',
                start=match.start(),
                end=match.end(),
            ))

    # Check date patterns
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            deadlines.append(Deadline(
                text=match.group(),
                deadline_type='date',
                start=match.start(),
                end=match.end(),
            ))

    # Deduplicate overlapping matches
    deadlines = _deduplicate_spans(deadlines)

    return deadlines


def extract_action_items(text: str) -> list[ActionItem]:
    """Extract action items and requests from email text.

    Args:
        text: Email body or subject text

    Returns:
        List of ActionItem objects
    """
    action_items = []

    # Check action patterns
    for pattern in ACTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            action_items.append(ActionItem(
                text=match.group(),
                action_type='request',
                start=match.start(),
                end=match.end(),
            ))

    # Check question patterns
    for pattern in QUESTION_PATTERNS:
        for match in re.finditer(pattern, text):
            action_items.append(ActionItem(
                text=match.group(),
                action_type='question',
                start=match.start(),
                end=match.end(),
            ))

    # Deduplicate overlapping matches
    action_items = _deduplicate_spans(action_items)

    return action_items


def _deduplicate_spans(items: list) -> list:
    """Remove overlapping spans, keeping longest matches."""
    if not items:
        return items

    # Sort by start position, then by length (longest first)
    sorted_items = sorted(items, key=lambda x: (x.start, -(x.end - x.start)))

    result = []
    last_end = -1

    for item in sorted_items:
        if item.start >= last_end:
            result.append(item)
            last_end = item.end

    return result


def extract_project_features(
    subject: str,
    body: str,
    *,
    subject_weight: float = 2.0,
) -> ProjectFeatures:
    """Extract all project-related features from an email.

    Args:
        subject: Email subject line
        body: Email body text
        subject_weight: Weight multiplier for subject matches (default 2.0)

    Returns:
        ProjectFeatures dataclass with extracted features
    """
    # Combine text for analysis
    combined_text = f"{subject}\n\n{body}"

    # Extract all components
    project_mentions = detect_project_mentions(combined_text)
    deadlines = detect_deadlines(combined_text)
    action_items = extract_action_items(combined_text)

    # Count questions separately
    question_count = sum(1 for a in action_items if a.action_type == 'question')

    # Calculate scores
    # Project score: based on mention count with diminishing returns
    project_mention_count = len(project_mentions)
    project_score = min(1.0, project_mention_count * 0.3)

    # Boost for subject mentions
    subject_mentions = detect_project_mentions(subject)
    if subject_mentions:
        project_score = min(1.0, project_score + 0.3 * subject_weight)

    # Urgency score: based on deadline/urgency markers
    deadline_count = len(deadlines)
    urgency_markers = sum(1 for d in deadlines if d.deadline_type == 'urgency')
    date_markers = sum(1 for d in deadlines if d.deadline_type == 'date')

    # Urgency words have higher weight than just dates
    urgency_score = min(1.0, urgency_markers * 0.4 + date_markers * 0.2)

    # Boost for subject urgency
    subject_deadlines = detect_deadlines(subject)
    subject_urgency = sum(1 for d in subject_deadlines if d.deadline_type == 'urgency')
    if subject_urgency:
        urgency_score = min(1.0, urgency_score + 0.3 * subject_weight)

    # Action score: based on action items and questions
    action_item_count = len(action_items)
    request_count = sum(1 for a in action_items if a.action_type == 'request')
    action_score = min(1.0, request_count * 0.2 + question_count * 0.15)

    return ProjectFeatures(
        project_mention_count=project_mention_count,
        deadline_count=deadline_count,
        action_item_count=action_item_count,
        question_count=question_count,
        project_score=project_score,
        urgency_score=urgency_score,
        action_score=action_score,
        project_mentions=project_mentions,
        deadlines=deadlines,
        action_items=action_items,
    )


def process_email_batch(
    emails: list[dict],
    *,
    subject_key: str = 'subject',
    body_key: str = 'body',
) -> list[ProjectFeatures]:
    """Process a batch of emails and extract project features.

    Args:
        emails: List of email dictionaries
        subject_key: Key for subject field in email dict
        body_key: Key for body field in email dict

    Returns:
        List of ProjectFeatures, one per email
    """
    results = []
    for email in emails:
        subject = email.get(subject_key, '')
        body = email.get(body_key, '')
        features = extract_project_features(subject, body)
        results.append(features)
    return results


if __name__ == '__main__':
    # Example usage
    sample_subject = "URGENT: Project Eagle Phase II - Action Required by EOD"
    sample_body = """
    Hi Team,

    Please review the attached proposal for Project Eagle Phase II.
    We need your approval by end of day today.

    Key action items:
    - Review budget estimates (Contract #12345)
    - Confirm timeline with Operations team
    - Send feedback to John by 5pm

    Let me know if you have any questions.

    Thanks,
    Jane
    """

    features = extract_project_features(sample_subject, sample_body)

    print("Project Features Extracted:")
    print(f"  Project mentions: {features.project_mention_count}")
    for m in features.project_mentions:
        print(f"    - {m.text}")

    print(f"  Deadlines: {features.deadline_count}")
    for d in features.deadlines:
        print(f"    - {d.text} ({d.deadline_type})")

    print(f"  Action items: {features.action_item_count}")
    for a in features.action_items:
        print(f"    - {a.text} ({a.action_type})")

    print(f"\nScores:")
    print(f"  Project score: {features.project_score:.2f}")
    print(f"  Urgency score: {features.urgency_score:.2f}")
    print(f"  Action score: {features.action_score:.2f}")

    print(f"\nFeature vector: {features.to_feature_vector()}")
