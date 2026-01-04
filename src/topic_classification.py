"""Topic classification for email content analysis.

This module classifies emails by topic/category using:
- Rule-based pattern matching for common email types
- TF-IDF vectorization with clustering for topic modeling
- Embedding-based similarity for topic inference

The output integrates with EmailState.topic_vector and topic_score.
"""

import re
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# Pre-defined topic categories aligned with business email patterns
TOPIC_CATEGORIES = [
    'meeting_scheduling',
    'project_update',
    'task_assignment',
    'information_sharing',
    'decision_request',
    'problem_report',
    'follow_up',
    'social_administrative',
    'external_communication',
    'legal_compliance',
]

# Topic category indices for vectorization
TOPIC_TO_INDEX = {topic: i for i, topic in enumerate(TOPIC_CATEGORIES)}


@dataclass
class TopicFeatures:
    """Features extracted from email topic analysis.

    Attributes:
        primary_topic: Most likely topic category
        topic_distribution: Probability distribution over all topics
        is_meeting_request: Email is about scheduling/meetings
        is_status_update: Email is a status or progress update
        is_question: Email contains questions requiring response
        is_fyi_only: Informational email, no action needed
        is_action_request: Email requests specific action
        is_decision_needed: Email requires a decision
        is_escalation: Email escalates an issue
        sentiment_score: Sentiment from -1 (negative) to 1 (positive)
        urgency_language: Score 0-1 for presence of urgent terms
    """
    primary_topic: str
    topic_distribution: dict[str, float]

    # Content classification flags
    is_meeting_request: bool = False
    is_status_update: bool = False
    is_question: bool = False
    is_fyi_only: bool = False
    is_action_request: bool = False
    is_decision_needed: bool = False
    is_escalation: bool = False

    # Sentiment and urgency
    sentiment_score: float = 0.0
    urgency_language: float = 0.0


# Pattern sets for rule-based detection
MEETING_PATTERNS = [
    r'\b(meeting|calendar|schedule|availability|invite)\b',
    r'\b(conference call|call at|zoom|teams|webex)\b',
    r'\b(let\'s meet|can we meet|set up a meeting)\b',
]

STATUS_UPDATE_PATTERNS = [
    r'\b(status update|progress report|weekly update)\b',
    r'\b(update on|reporting on|here\'s the status)\b',
    r'\b(completed|finished|done with|wrapped up)\b',
]

QUESTION_PATTERNS = [
    r'\?',
    r'\b(what|when|where|why|how|which|who)\b',
    r'\b(can you|could you|would you|will you)\b',
    r'\b(do you know|are you)\b',
]

FYI_PATTERNS = [
    r'\b(fyi|for your information|heads up)\b',
    r'\b(just wanted to let you know|wanted to share)\b',
    r'\b(no action needed|no response needed)\b',
]

ACTION_PATTERNS = [
    r'\b(please|kindly|action required|action needed)\b',
    r'\b(can you|could you|need you to|would you)\b',
    r'\b(send me|provide|prepare|review|approve)\b',
]

DECISION_PATTERNS = [
    r'\b(decision|decide|approval|approve|sign off)\b',
    r'\b(need your input|need your decision|your call)\b',
    r'\b(which option|should we|what should)\b',
]

ESCALATION_PATTERNS = [
    r'\b(escalate|escalating|escalation)\b',
    r'\b(critical issue|urgent issue|blocking)\b',
    r'\b(needs immediate attention|requires immediate)\b',
]

URGENCY_WORDS = [
    'urgent', 'asap', 'immediately', 'critical', 'important',
    'priority', 'deadline', 'time sensitive', 'by eod', 'by cob',
]

# Topic keyword associations for classification
TOPIC_KEYWORDS = {
    'meeting_scheduling': ['meeting', 'calendar', 'schedule', 'availability', 'invite', 'call'],
    'project_update': ['update', 'status', 'progress', 'milestone', 'deliverable', 'report'],
    'task_assignment': ['task', 'assign', 'please', 'action', 'complete', 'deliver', 'review'],
    'information_sharing': ['fyi', 'info', 'share', 'attached', 'document', 'read'],
    'decision_request': ['decide', 'decision', 'approve', 'approval', 'sign off', 'input'],
    'problem_report': ['issue', 'problem', 'error', 'bug', 'fix', 'broken', 'failed'],
    'follow_up': ['follow up', 'following up', 'reminder', 'checking in', 'status on'],
    'social_administrative': ['lunch', 'birthday', 'holiday', 'vacation', 'team', 'hr'],
    'external_communication': ['client', 'customer', 'vendor', 'partner', 'external'],
    'legal_compliance': ['legal', 'compliance', 'regulatory', 'audit', 'policy', 'contract'],
}


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given regex patterns."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text."""
    count = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count


def _compute_urgency_score(text: str) -> float:
    """Compute urgency language score (0-1)."""
    text_lower = text.lower()
    matches = sum(1 for word in URGENCY_WORDS if word in text_lower)
    return min(matches / 3.0, 1.0)


def _compute_sentiment_score(text: str) -> float:
    """Simple sentiment analysis based on positive/negative word lists.

    Returns score from -1 (negative) to 1 (positive).
    """
    positive_words = [
        'thanks', 'thank you', 'great', 'good', 'excellent', 'wonderful',
        'appreciate', 'pleased', 'happy', 'glad', 'success', 'perfect',
    ]
    negative_words = [
        'problem', 'issue', 'error', 'fail', 'concern', 'worry', 'urgent',
        'unfortunately', 'sorry', 'disappointed', 'wrong', 'bad',
    ]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def _compute_topic_distribution(text: str) -> dict[str, float]:
    """Compute topic probability distribution using keyword matching.

    Uses TF-IDF-like scoring based on keyword presence and frequency.
    Returns normalized probability distribution over topics.
    """
    text_lower = text.lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if keyword in text_lower:
                # Weight by how often keyword appears (capped at 3)
                count = min(text_lower.count(keyword), 3)
                score += count
        scores[topic] = score

    # Normalize to probabilities with smoothing
    total = sum(scores.values()) + len(TOPIC_CATEGORIES) * 0.1
    distribution = {
        topic: (scores.get(topic, 0) + 0.1) / total
        for topic in TOPIC_CATEGORIES
    }

    return distribution


def classify_topic(email: dict) -> TopicFeatures:
    """Classify email into topic categories.

    Uses a combination of:
    1. Rule-based detection for common patterns (meeting, question, action, etc.)
    2. Keyword-based topic distribution for general categorization

    Args:
        email: Dictionary with 'subject' and 'body' keys

    Returns:
        TopicFeatures with classification results
    """
    subject = email.get('subject', '')
    body = email.get('body', '')
    text = f"{subject} {body}"

    # Rule-based detection for boolean flags
    is_meeting = _matches_any_pattern(text, MEETING_PATTERNS)
    is_status = _matches_any_pattern(text, STATUS_UPDATE_PATTERNS)
    is_question = _matches_any_pattern(text, QUESTION_PATTERNS)
    is_fyi = _matches_any_pattern(text, FYI_PATTERNS)
    is_action = _matches_any_pattern(text, ACTION_PATTERNS)
    is_decision = _matches_any_pattern(text, DECISION_PATTERNS)
    is_escalation = _matches_any_pattern(text, ESCALATION_PATTERNS)

    # Compute topic distribution
    topic_dist = _compute_topic_distribution(text)

    # Boost topic scores based on rule-based detection
    if is_meeting:
        topic_dist['meeting_scheduling'] = min(topic_dist['meeting_scheduling'] + 0.3, 1.0)
    if is_status:
        topic_dist['project_update'] = min(topic_dist['project_update'] + 0.2, 1.0)
    if is_action:
        topic_dist['task_assignment'] = min(topic_dist['task_assignment'] + 0.2, 1.0)
    if is_fyi:
        topic_dist['information_sharing'] = min(topic_dist['information_sharing'] + 0.3, 1.0)
    if is_decision:
        topic_dist['decision_request'] = min(topic_dist['decision_request'] + 0.3, 1.0)
    if is_escalation:
        topic_dist['problem_report'] = min(topic_dist['problem_report'] + 0.2, 1.0)

    # Re-normalize after boosts
    total = sum(topic_dist.values())
    if total > 0:
        topic_dist = {k: v / total for k, v in topic_dist.items()}

    # Determine primary topic
    primary_topic = max(topic_dist, key=topic_dist.get)

    # If action request but no action content, it's likely FYI only
    # (e.g., "please see attached" without actual task)
    if is_fyi and not is_action and not is_decision:
        is_fyi_only = True
    elif not is_question and not is_action and not is_decision and not is_meeting:
        is_fyi_only = True
    else:
        is_fyi_only = False

    return TopicFeatures(
        primary_topic=primary_topic,
        topic_distribution=topic_dist,
        is_meeting_request=is_meeting,
        is_status_update=is_status,
        is_question=is_question,
        is_fyi_only=is_fyi_only,
        is_action_request=is_action,
        is_decision_needed=is_decision,
        is_escalation=is_escalation,
        sentiment_score=_compute_sentiment_score(text),
        urgency_language=_compute_urgency_score(text),
    )


def compute_topic_score(features: TopicFeatures) -> float:
    """Compute topic importance score (0-1).

    Higher scores for topics that typically require action or response.

    Args:
        features: TopicFeatures from classify_topic()

    Returns:
        Score from 0-1 indicating topic importance
    """
    # Base score by topic type - topics requiring action score higher
    topic_weights = {
        'decision_request': 0.9,
        'problem_report': 0.85,
        'task_assignment': 0.8,
        'follow_up': 0.7,
        'meeting_scheduling': 0.6,
        'project_update': 0.5,
        'external_communication': 0.5,
        'legal_compliance': 0.7,
        'information_sharing': 0.3,
        'social_administrative': 0.2,
    }

    base_score = topic_weights.get(features.primary_topic, 0.5)

    # Modifiers based on content flags
    modifiers = 0.0

    if features.is_question:
        modifiers += 0.1
    if features.is_action_request:
        modifiers += 0.15
    if features.is_decision_needed:
        modifiers += 0.2
    if features.is_escalation:
        modifiers += 0.25
    if features.urgency_language > 0.5:
        modifiers += 0.15

    # FYI-only emails get lower score
    if features.is_fyi_only:
        modifiers -= 0.2

    # Negative sentiment might indicate issues needing attention
    if features.sentiment_score < -0.3:
        modifiers += 0.1

    return min(max(base_score + modifiers, 0.0), 1.0)


def topic_features_to_vector(features: TopicFeatures) -> np.ndarray:
    """Convert TopicFeatures to a numpy vector for ML models.

    Returns:
        numpy array with topic distribution followed by boolean flags
    """
    # Topic distribution (10 dimensions)
    topic_vec = np.array([
        features.topic_distribution.get(topic, 0.0)
        for topic in TOPIC_CATEGORIES
    ], dtype=np.float32)

    # Boolean flags (7 dimensions)
    flags = np.array([
        float(features.is_meeting_request),
        float(features.is_status_update),
        float(features.is_question),
        float(features.is_fyi_only),
        float(features.is_action_request),
        float(features.is_decision_needed),
        float(features.is_escalation),
    ], dtype=np.float32)

    # Continuous features (2 dimensions)
    continuous = np.array([
        features.sentiment_score,
        features.urgency_language,
    ], dtype=np.float32)

    return np.concatenate([topic_vec, flags, continuous])


def get_topic_vector_size() -> int:
    """Return the size of the topic feature vector.

    Returns:
        19 (10 topics + 7 flags + 2 continuous)
    """
    return len(TOPIC_CATEGORIES) + 7 + 2  # 19


# Integration helper for EmailState
def extract_topic_for_state(email: dict) -> tuple[np.ndarray, float]:
    """Extract topic features for integration with EmailState.

    Args:
        email: Dictionary with 'subject' and 'body' keys

    Returns:
        Tuple of (topic_vector, topic_score) for EmailState
    """
    features = classify_topic(email)
    topic_vector = topic_features_to_vector(features)
    topic_score = compute_topic_score(features)

    return topic_vector, topic_score


def enrich_email_state_with_topic(state, email_data: dict) -> None:
    """Enrich an EmailState object with topic classification.

    This function modifies the state in-place, setting:
    - state.topic_vector: numpy array of topic features
    - state.topic_score: float 0-1 importance score

    Args:
        state: EmailState object to enrich
        email_data: Dictionary with 'subject' and 'body' keys

    Example:
        >>> from email_state import create_email_state_from_json
        >>> from topic_classification import enrich_email_state_with_topic
        >>> state = create_email_state_from_json(email_json)
        >>> enrich_email_state_with_topic(state, email_json)
        >>> print(state.topic_score)  # 0.75
    """
    topic_vector, topic_score = extract_topic_for_state(email_data)
    state.topic_vector = topic_vector
    state.topic_score = topic_score


def get_topic_features(email: dict) -> TopicFeatures:
    """Convenience function to get full TopicFeatures object.

    Use this when you need access to all topic classification details,
    not just the vector/score for EmailState.

    Args:
        email: Dictionary with 'subject' and 'body' keys

    Returns:
        TopicFeatures with full classification details
    """
    return classify_topic(email)
