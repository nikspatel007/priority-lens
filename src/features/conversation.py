#!/usr/bin/env python3
"""Conversation classification for people emails.

TYPE-001: Classifies email conversations into categories:
- LONG_RUNNING: Extended email chains (5+ messages, multi-day)
- SHORT_EXCHANGE: Quick 2-4 message exchanges
- SINGLE_MESSAGE: One-off emails, no response expected
- BROADCAST: Mass emails to many recipients

Uses traditional features for classification and LLM for topic drift detection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


class ConversationType(str, Enum):
    """Email conversation type classification."""
    LONG_RUNNING = "long_running"      # Extended chains: 5+ messages, spans days
    SHORT_EXCHANGE = "short_exchange"  # Quick exchanges: 2-4 messages
    SINGLE_MESSAGE = "single_message"  # One-off: no response expected
    BROADCAST = "broadcast"            # Mass emails: newsletters, announcements


@dataclass
class ThreadContext:
    """Thread-level context for conversation classification.

    This aggregates information about the thread that an email belongs to.
    """
    thread_id: str
    message_count: int = 1
    participant_count: int = 1
    participant_emails: list[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    avg_response_time_seconds: Optional[float] = None

    # User participation
    user_message_count: int = 0
    user_reply_count: int = 0

    # Message position
    position_in_thread: int = 0  # 0-indexed position of current email
    is_first_message: bool = True
    is_latest_message: bool = True

    # Recipient patterns (from the current email)
    to_count: int = 0
    cc_count: int = 0
    bcc_count: int = 0
    total_recipients: int = 0


@dataclass
class TopicDriftInfo:
    """Information about topic drift within a thread (LLM-extracted)."""
    has_topic_drift: bool = False
    drift_confidence: float = 0.0  # 0-1, how confident we are drift occurred
    original_topic: Optional[str] = None
    current_topic: Optional[str] = None
    drift_point_index: Optional[int] = None  # Message index where drift occurred
    topic_segments: list[dict] = field(default_factory=list)  # [{start, end, topic}]


@dataclass
class ConversationFeatures:
    """Features for conversation classification."""
    # Classification result
    conversation_type: ConversationType
    type_confidence: float  # 0-1

    # Thread metrics
    thread_id: str
    message_count: int
    participant_count: int
    duration_hours: Optional[float] = None
    avg_response_time_hours: Optional[float] = None

    # Broadcast indicators
    is_broadcast: bool = False
    broadcast_confidence: float = 0.0
    recipient_count: int = 0
    cc_to_ratio: float = 0.0  # CC / (To + CC)

    # Long-running indicators
    is_long_running: bool = False
    spans_multiple_days: bool = False
    has_deep_thread: bool = False

    # User engagement
    user_participation_rate: float = 0.0  # user messages / total messages
    user_initiated: bool = False

    # Topic drift (LLM-extracted)
    topic_drift: Optional[TopicDriftInfo] = None

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to numerical vector for ML pipeline.

        Returns 15-dimensional vector.
        """
        # Encode conversation type as one-hot (4 dims)
        type_encoding = [0.0, 0.0, 0.0, 0.0]
        type_map = {
            ConversationType.LONG_RUNNING: 0,
            ConversationType.SHORT_EXCHANGE: 1,
            ConversationType.SINGLE_MESSAGE: 2,
            ConversationType.BROADCAST: 3,
        }
        type_encoding[type_map[self.conversation_type]] = 1.0

        values = [
            # Type encoding (4)
            *type_encoding,
            # Type confidence (1)
            self.type_confidence,
            # Thread metrics (4)
            min(self.message_count, 50) / 50.0,
            min(self.participant_count, 20) / 20.0,
            min(self.duration_hours or 0, 720) / 720.0,  # Cap at 30 days
            min(self.avg_response_time_hours or 24, 168) / 168.0,  # Cap at 1 week
            # Broadcast indicators (2)
            self.broadcast_confidence,
            self.cc_to_ratio,
            # Engagement (2)
            self.user_participation_rate,
            1.0 if self.user_initiated else 0.0,
            # Topic drift (2)
            1.0 if (self.topic_drift and self.topic_drift.has_topic_drift) else 0.0,
            self.topic_drift.drift_confidence if self.topic_drift else 0.0,
        ]

        if HAS_NUMPY:
            return np.array(values, dtype=np.float32)
        return values


# Thresholds for classification
BROADCAST_RECIPIENT_THRESHOLD = 10  # 10+ recipients suggests broadcast
BROADCAST_CC_RATIO_THRESHOLD = 0.7  # High CC ratio suggests broadcast
LONG_RUNNING_MESSAGE_THRESHOLD = 5  # 5+ messages
LONG_RUNNING_DURATION_HOURS = 48    # 48+ hours


def classify_conversation(
    email: dict,
    thread_context: Optional[ThreadContext] = None,
    *,
    user_email: str = "",
) -> ConversationFeatures:
    """Classify an email's conversation type based on thread context.

    Args:
        email: Email dictionary with standard fields
        thread_context: Optional thread context for richer classification
        user_email: User's email for computing participation

    Returns:
        ConversationFeatures with classification and metrics
    """
    # Extract recipient info from current email
    to_field = email.get("to", "") or email.get("to_emails", [])
    cc_field = email.get("cc", "") or email.get("cc_emails", [])

    if isinstance(to_field, str):
        to_count = len([x for x in to_field.split(",") if x.strip()])
    else:
        to_count = len(to_field) if to_field else 0

    if isinstance(cc_field, str):
        cc_count = len([x for x in cc_field.split(",") if x.strip()])
    else:
        cc_count = len(cc_field) if cc_field else 0

    total_recipients = to_count + cc_count
    cc_to_ratio = cc_count / max(total_recipients, 1)

    # Get thread metrics
    if thread_context:
        message_count = thread_context.message_count
        participant_count = thread_context.participant_count
        duration_hours = None
        if thread_context.duration_seconds is not None:
            duration_hours = thread_context.duration_seconds / 3600.0
        avg_response_hours = None
        if thread_context.avg_response_time_seconds is not None:
            avg_response_hours = thread_context.avg_response_time_seconds / 3600.0
        thread_id = thread_context.thread_id

        # User participation
        if message_count > 0:
            user_participation = thread_context.user_message_count / message_count
        else:
            user_participation = 0.0
        user_initiated = thread_context.is_first_message and thread_context.user_message_count > 0
    else:
        # Single email context - extract what we can
        message_count = 1
        participant_count = 1 + total_recipients
        duration_hours = None
        avg_response_hours = None
        thread_id = email.get("thread_id", email.get("message_id", "unknown"))
        user_participation = 1.0 if user_email else 0.0
        user_initiated = False

    # Classification logic
    conversation_type, confidence = _classify_type(
        message_count=message_count,
        participant_count=participant_count,
        duration_hours=duration_hours,
        total_recipients=total_recipients,
        cc_to_ratio=cc_to_ratio,
    )

    # Compute indicator flags
    is_broadcast = conversation_type == ConversationType.BROADCAST
    broadcast_confidence = _compute_broadcast_confidence(
        total_recipients, cc_to_ratio, message_count
    )

    is_long_running = conversation_type == ConversationType.LONG_RUNNING
    spans_multiple_days = duration_hours is not None and duration_hours >= 24
    has_deep_thread = message_count >= LONG_RUNNING_MESSAGE_THRESHOLD

    return ConversationFeatures(
        conversation_type=conversation_type,
        type_confidence=confidence,
        thread_id=thread_id,
        message_count=message_count,
        participant_count=participant_count,
        duration_hours=duration_hours,
        avg_response_time_hours=avg_response_hours,
        is_broadcast=is_broadcast,
        broadcast_confidence=broadcast_confidence,
        recipient_count=total_recipients,
        cc_to_ratio=cc_to_ratio,
        is_long_running=is_long_running,
        spans_multiple_days=spans_multiple_days,
        has_deep_thread=has_deep_thread,
        user_participation_rate=user_participation,
        user_initiated=user_initiated,
        topic_drift=None,  # Set via LLM extraction
    )


def _classify_type(
    message_count: int,
    participant_count: int,
    duration_hours: Optional[float],
    total_recipients: int,
    cc_to_ratio: float,
) -> tuple[ConversationType, float]:
    """Determine conversation type with confidence score."""
    scores = {
        ConversationType.LONG_RUNNING: 0.0,
        ConversationType.SHORT_EXCHANGE: 0.0,
        ConversationType.SINGLE_MESSAGE: 0.0,
        ConversationType.BROADCAST: 0.0,
    }

    # BROADCAST: High recipient count or CC ratio
    if total_recipients >= BROADCAST_RECIPIENT_THRESHOLD:
        scores[ConversationType.BROADCAST] += 0.6
    if cc_to_ratio >= BROADCAST_CC_RATIO_THRESHOLD:
        scores[ConversationType.BROADCAST] += 0.3
    if message_count == 1 and total_recipients >= 5:
        scores[ConversationType.BROADCAST] += 0.2

    # SINGLE_MESSAGE: Only one message in thread
    if message_count == 1:
        # Could be single or broadcast
        if scores[ConversationType.BROADCAST] < 0.5:
            scores[ConversationType.SINGLE_MESSAGE] += 0.8

    # LONG_RUNNING: Many messages, spans time
    if message_count >= LONG_RUNNING_MESSAGE_THRESHOLD:
        scores[ConversationType.LONG_RUNNING] += 0.5
    if duration_hours is not None and duration_hours >= LONG_RUNNING_DURATION_HOURS:
        scores[ConversationType.LONG_RUNNING] += 0.3
    if participant_count >= 3 and message_count >= 3:
        scores[ConversationType.LONG_RUNNING] += 0.2

    # SHORT_EXCHANGE: 2-4 messages
    if 2 <= message_count <= 4:
        scores[ConversationType.SHORT_EXCHANGE] += 0.6
        if duration_hours is not None and duration_hours <= 24:
            scores[ConversationType.SHORT_EXCHANGE] += 0.2

    # Find winner
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Normalize confidence
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = best_score / total_score
    else:
        confidence = 0.5

    # Handle edge case: no strong signal
    if best_score == 0:
        if message_count == 1:
            return ConversationType.SINGLE_MESSAGE, 0.5
        elif message_count <= 4:
            return ConversationType.SHORT_EXCHANGE, 0.5
        else:
            return ConversationType.LONG_RUNNING, 0.5

    return best_type, min(1.0, confidence)


def _compute_broadcast_confidence(
    total_recipients: int,
    cc_to_ratio: float,
    message_count: int,
) -> float:
    """Compute confidence that this is a broadcast email."""
    confidence = 0.0

    # High recipient count
    if total_recipients >= BROADCAST_RECIPIENT_THRESHOLD:
        confidence += 0.5 * min(total_recipients / 20, 1.0)
    elif total_recipients >= 5:
        confidence += 0.2

    # High CC ratio
    if cc_to_ratio >= BROADCAST_CC_RATIO_THRESHOLD:
        confidence += 0.3
    elif cc_to_ratio >= 0.5:
        confidence += 0.15

    # Single message (broadcasts rarely get replies)
    if message_count == 1:
        confidence += 0.1
    elif message_count > 3:
        confidence -= 0.2  # Likely not a broadcast if there's discussion

    return max(0.0, min(1.0, confidence))


def compute_conversation_score(features: ConversationFeatures) -> float:
    """Compute a priority score adjustment based on conversation type.

    Returns a 0-1 score where:
    - Higher for long-running threads (ongoing important discussions)
    - Lower for broadcasts (less personal urgency)
    - Medium for short exchanges and single messages
    """
    base_scores = {
        ConversationType.LONG_RUNNING: 0.7,
        ConversationType.SHORT_EXCHANGE: 0.5,
        ConversationType.SINGLE_MESSAGE: 0.4,
        ConversationType.BROADCAST: 0.2,
    }

    score = base_scores[features.conversation_type]

    # Adjust based on user participation
    if features.user_participation_rate > 0.3:
        score += 0.15  # User is actively engaged
    if features.user_initiated:
        score += 0.1  # User started this conversation

    # Topic drift can indicate important evolving discussions
    if features.topic_drift and features.topic_drift.has_topic_drift:
        score += 0.1 * features.topic_drift.drift_confidence

    return max(0.0, min(1.0, score))


# ============================================================================
# Topic Drift Detection (LLM-based)
# ============================================================================

TOPIC_DRIFT_SYSTEM_PROMPT = """You are an email thread analyst. Analyze topic drift within email threads.

Given a sequence of emails from the same thread, identify:
1. Whether the topic has shifted from the original subject
2. The original topic and current topic if drift occurred
3. Confidence level (0-1) in your assessment

Output ONLY valid JSON:
{
  "has_topic_drift": true or false,
  "drift_confidence": 0.0 to 1.0,
  "original_topic": "brief description" or null,
  "current_topic": "brief description" or null,
  "drift_point_index": message index where drift occurred or null,
  "topic_segments": [{"start": 0, "end": 2, "topic": "..."}, ...]
}

Rules:
- has_topic_drift: true if conversation shifted to substantially different topic
- Minor elaborations on same topic are NOT drift
- Drift means the main focus/purpose of conversation changed
- topic_segments: list segments if multiple topics discussed
"""


def format_thread_for_llm(emails: list[dict], max_emails: int = 10) -> str:
    """Format thread emails for LLM topic drift analysis.

    Args:
        emails: List of email dicts sorted by date
        max_emails: Maximum emails to include (sample evenly if more)

    Returns:
        Formatted string for LLM prompt
    """
    if len(emails) > max_emails:
        # Sample evenly across thread
        indices = [int(i * len(emails) / max_emails) for i in range(max_emails)]
        emails = [emails[i] for i in indices]

    parts = []
    for i, email in enumerate(emails):
        subject = email.get("subject", "")
        body = email.get("body", email.get("body_text", ""))[:500]
        sender = email.get("from", email.get("from_email", ""))

        parts.append(f"[Email {i}]\nFrom: {sender}\nSubject: {subject}\n\n{body}\n")

    return "\n---\n".join(parts)


async def detect_topic_drift_async(
    emails: list[dict],
    client: "AsyncAnthropic",
    *,
    model: str = "claude-haiku-4-5-20251001",
) -> TopicDriftInfo:
    """Detect topic drift in a thread using LLM.

    Args:
        emails: List of emails in thread, sorted by date
        client: AsyncAnthropic client
        model: Model to use

    Returns:
        TopicDriftInfo with drift analysis
    """
    import json

    if len(emails) < 2:
        return TopicDriftInfo(has_topic_drift=False)

    thread_content = format_thread_for_llm(emails)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            system=TOPIC_DRIFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": thread_content}],
        )

        response_text = response.content[0].text.strip()
        data = _parse_json_response(response_text)

        if data:
            return TopicDriftInfo(
                has_topic_drift=data.get("has_topic_drift", False),
                drift_confidence=float(data.get("drift_confidence", 0.0)),
                original_topic=data.get("original_topic"),
                current_topic=data.get("current_topic"),
                drift_point_index=data.get("drift_point_index"),
                topic_segments=data.get("topic_segments", []),
            )
    except Exception:
        pass

    return TopicDriftInfo(has_topic_drift=False)


def detect_topic_drift_sync(
    emails: list[dict],
    client: "Anthropic",
    *,
    model: str = "claude-haiku-4-5-20251001",
) -> TopicDriftInfo:
    """Synchronous version of topic drift detection.

    Args:
        emails: List of emails in thread, sorted by date
        client: Anthropic client (sync)
        model: Model to use

    Returns:
        TopicDriftInfo with drift analysis
    """
    import json

    if len(emails) < 2:
        return TopicDriftInfo(has_topic_drift=False)

    thread_content = format_thread_for_llm(emails)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=TOPIC_DRIFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": thread_content}],
        )

        response_text = response.content[0].text.strip()
        data = _parse_json_response(response_text)

        if data:
            return TopicDriftInfo(
                has_topic_drift=data.get("has_topic_drift", False),
                drift_confidence=float(data.get("drift_confidence", 0.0)),
                original_topic=data.get("original_topic"),
                current_topic=data.get("current_topic"),
                drift_point_index=data.get("drift_point_index"),
                topic_segments=data.get("topic_segments", []),
            )
    except Exception:
        pass

    return TopicDriftInfo(has_topic_drift=False)


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM response, handling markdown wrapping."""
    import json

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


# ============================================================================
# Batch Processing
# ============================================================================

def classify_batch(
    emails: list[dict],
    thread_contexts: Optional[dict[str, ThreadContext]] = None,
    *,
    user_email: str = "",
) -> list[ConversationFeatures]:
    """Classify conversation type for a batch of emails.

    Args:
        emails: List of email dictionaries
        thread_contexts: Optional mapping of thread_id -> ThreadContext
        user_email: User's email address

    Returns:
        List of ConversationFeatures, one per email
    """
    results = []
    for email in emails:
        thread_id = email.get("thread_id", email.get("message_id", ""))
        context = thread_contexts.get(thread_id) if thread_contexts else None
        features = classify_conversation(email, context, user_email=user_email)
        results.append(features)
    return results


def build_thread_contexts(
    emails: list[dict],
    user_email: str = "",
) -> dict[str, ThreadContext]:
    """Build thread contexts from a list of emails.

    Groups emails by thread_id and computes aggregated context.

    Args:
        emails: List of email dictionaries
        user_email: User's email address

    Returns:
        Mapping of thread_id -> ThreadContext
    """
    from collections import defaultdict

    # Group by thread_id
    thread_emails: dict[str, list[dict]] = defaultdict(list)
    for email in emails:
        thread_id = email.get("thread_id", email.get("message_id", ""))
        if thread_id:
            thread_emails[thread_id].append(email)

    contexts = {}
    user_email_lower = user_email.lower().strip()

    for thread_id, thread_email_list in thread_emails.items():
        # Sort by date
        sorted_emails = sorted(
            thread_email_list,
            key=lambda e: e.get("date") or e.get("date_parsed") or datetime.min
        )

        # Collect participants
        participants = set()
        for email in sorted_emails:
            sender = email.get("from", email.get("from_email", "")).lower()
            if sender:
                participants.add(sender)
            for to in email.get("to_emails", []):
                if isinstance(to, str):
                    participants.add(to.lower())

        # Compute timing
        dates = []
        for email in sorted_emails:
            date = email.get("date") or email.get("date_parsed")
            if isinstance(date, datetime):
                dates.append(date)
            elif isinstance(date, str):
                try:
                    from email.utils import parsedate_to_datetime
                    dates.append(parsedate_to_datetime(date))
                except Exception:
                    pass

        started_at = min(dates) if dates else None
        last_activity = max(dates) if dates else None
        duration_seconds = None
        if started_at and last_activity and len(dates) > 1:
            duration_seconds = (last_activity - started_at).total_seconds()

        # Compute avg response time
        avg_response_seconds = None
        if len(dates) > 1:
            response_times = []
            for i in range(1, len(dates)):
                delta = (dates[i] - dates[i - 1]).total_seconds()
                if delta > 0:
                    response_times.append(delta)
            if response_times:
                avg_response_seconds = sum(response_times) / len(response_times)

        # User participation
        user_message_count = 0
        user_reply_count = 0
        for email in sorted_emails:
            sender = email.get("from", email.get("from_email", "")).lower()
            if sender == user_email_lower:
                user_message_count += 1
                if email.get("in_reply_to"):
                    user_reply_count += 1

        contexts[thread_id] = ThreadContext(
            thread_id=thread_id,
            message_count=len(sorted_emails),
            participant_count=len(participants),
            participant_emails=list(participants),
            started_at=started_at,
            last_activity=last_activity,
            duration_seconds=duration_seconds,
            avg_response_time_seconds=avg_response_seconds,
            user_message_count=user_message_count,
            user_reply_count=user_reply_count,
        )

    return contexts


if __name__ == "__main__":
    # Example usage
    sample_emails = [
        {
            "message_id": "msg1",
            "thread_id": "thread1",
            "from": "alice@example.com",
            "to": "bob@example.com",
            "cc": "",
            "subject": "Project update",
            "body": "Here's the latest on the project...",
            "date": datetime(2024, 1, 1, 10, 0, 0),
        },
        {
            "message_id": "msg2",
            "thread_id": "thread1",
            "from": "bob@example.com",
            "to": "alice@example.com",
            "cc": "",
            "subject": "Re: Project update",
            "body": "Thanks for the update. One question...",
            "date": datetime(2024, 1, 1, 11, 0, 0),
            "in_reply_to": "msg1",
        },
        {
            "message_id": "msg3",
            "thread_id": "thread2",
            "from": "newsletter@company.com",
            "to": "bob@example.com, alice@example.com, charlie@example.com",
            "cc": "team@example.com, all@example.com, managers@example.com",
            "subject": "Weekly Newsletter",
            "body": "This week's updates...",
            "date": datetime(2024, 1, 2, 9, 0, 0),
        },
    ]

    # Build thread contexts
    contexts = build_thread_contexts(sample_emails, user_email="bob@example.com")
    print("Thread Contexts Built:")
    for tid, ctx in contexts.items():
        print(f"  {tid}: {ctx.message_count} messages, {ctx.participant_count} participants")
    print()

    # Classify conversations
    for email in sample_emails:
        thread_id = email.get("thread_id", "")
        context = contexts.get(thread_id)
        features = classify_conversation(email, context, user_email="bob@example.com")

        print(f"Email: {email['subject']}")
        print(f"  Type: {features.conversation_type.value}")
        print(f"  Confidence: {features.type_confidence:.2f}")
        print(f"  Recipients: {features.recipient_count}")
        print(f"  Is Broadcast: {features.is_broadcast}")
        print(f"  Score: {compute_conversation_score(features):.2f}")
        print()
