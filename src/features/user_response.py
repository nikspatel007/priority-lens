#!/usr/bin/env python3
"""User response history tracking per sender.

Computes user's reply rate and average response time for each sender.
Supports both SurrealDB queries and in-memory computation from email lists.

Output features:
- user_replied_to_sender_rate: Rate at which user replies to emails from this sender (0-1)
- avg_response_time_hours: User's average response time to this sender in hours
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional


def normalize_email(email_addr: str) -> str:
    """Extract and normalize email address from a From/To header."""
    if not email_addr:
        return ""

    # Extract email from angle brackets
    match = re.search(r'<([^>]+)>', email_addr)
    if match:
        email_addr = match.group(1)

    return email_addr.strip().lower().strip('<>"\'')


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse email date string to datetime."""
    if not date_str:
        return None

    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        for fmt in [
            '%a, %d %b %Y %H:%M:%S %z',
            '%d %b %Y %H:%M:%S %z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
        ]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
    return None


def extract_thread_id(subject: str) -> str:
    """Extract normalized thread ID from subject by removing Re:/Fwd: prefixes."""
    if not subject:
        return ""

    # Remove Re:, Fwd:, Fw: prefixes (case insensitive, multiple)
    cleaned = re.sub(r'^(?:re|fwd?|fw):\s*', '', subject.strip(), flags=re.IGNORECASE)
    while cleaned != subject:
        subject = cleaned
        cleaned = re.sub(r'^(?:re|fwd?|fw):\s*', '', subject.strip(), flags=re.IGNORECASE)

    # Normalize whitespace and lowercase
    return ' '.join(cleaned.lower().split())


@dataclass
class UserResponse:
    """User response history for a specific sender.

    Attributes:
        user: Normalized user email address (the one who responds)
        sender: Normalized sender email address (the one being responded to)
        reference_time: Time from which history is calculated
        user_replied_to_sender_rate: Rate of replies (0-1)
        avg_response_time_hours: Average response time in hours
        total_emails_from_sender: Total emails received from sender
        total_responses_to_sender: Total responses user sent to sender
        response_times: List of individual response times (for median/percentile)
    """
    user: str
    sender: str
    reference_time: datetime
    user_replied_to_sender_rate: float = 0.0
    avg_response_time_hours: Optional[float] = None
    total_emails_from_sender: int = 0
    total_responses_to_sender: int = 0
    response_times: list[float] = None

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'user': self.user,
            'sender': self.sender,
            'reference_time': self.reference_time.isoformat(),
            'user_replied_to_sender_rate': self.user_replied_to_sender_rate,
            'avg_response_time_hours': self.avg_response_time_hours,
            'total_emails_from_sender': self.total_emails_from_sender,
            'total_responses_to_sender': self.total_responses_to_sender,
        }

    def to_feature_vector(self) -> list[float]:
        """Convert to feature vector for ML pipeline.

        Returns 2-dimensional vector:
        - user_replied_to_sender_rate (0-1, already normalized)
        - avg_response_time_hours (normalized by 168h = 1 week)
        """
        return [
            self.user_replied_to_sender_rate,
            (self.avg_response_time_hours or 24.0) / 168.0,  # Normalize by 1 week
        ]

    @property
    def median_response_time_hours(self) -> Optional[float]:
        """Median response time in hours."""
        if not self.response_times:
            return None
        sorted_times = sorted(self.response_times)
        mid = len(sorted_times) // 2
        if len(sorted_times) % 2 == 0:
            return (sorted_times[mid - 1] + sorted_times[mid]) / 2
        return sorted_times[mid]


@dataclass
class _EmailRecord:
    """Internal record for indexing emails."""
    message_id: str
    sender: str
    recipients: set[str]
    timestamp: datetime
    thread_id: str
    in_reply_to: Optional[str]
    references: list[str]


class UserResponseIndex:
    """Index for efficient user response history lookups from email lists.

    Pre-indexes emails and detects response patterns for fast repeated lookups.
    """

    def __init__(self, emails: list[dict]):
        """Build index from email list.

        Args:
            emails: List of email dictionaries with 'from', 'to', 'date',
                   'subject', 'message_id', 'in_reply_to', 'references' fields
        """
        # All indexed emails
        self._emails: list[_EmailRecord] = []

        # Index by message_id for reply lookups
        self._by_message_id: dict[str, _EmailRecord] = {}

        # Index by thread_id for thread-based response detection
        self._by_thread_id: dict[str, list[_EmailRecord]] = defaultdict(list)

        # Emails received by user from sender: (user, sender) -> list of timestamps
        self._received: dict[tuple[str, str], list[datetime]] = defaultdict(list)

        # Response events: (user, sender) -> list of (original_time, response_time)
        self._responses: dict[tuple[str, str], list[tuple[datetime, datetime]]] = defaultdict(list)

        self._build_index(emails)
        self._detect_responses()

    def _build_index(self, emails: list[dict]) -> None:
        """Build indices from email list."""
        for email in emails:
            sender = normalize_email(email.get('from', ''))
            if not sender:
                continue

            email_date = parse_date(email.get('date', ''))
            if email_date is None:
                continue

            # Make date timezone-naive for consistent comparison
            if email_date.tzinfo is not None:
                email_date = email_date.replace(tzinfo=None)

            # Collect recipients
            to_emails = email.get('to', '') or ''
            cc_emails = email.get('cc', '') or ''

            recipients = set()
            if isinstance(to_emails, list):
                recipients.update(normalize_email(e) for e in to_emails)
            else:
                recipients.update(normalize_email(e) for e in re.split(r'[,;]', to_emails) if e.strip())

            if isinstance(cc_emails, list):
                recipients.update(normalize_email(e) for e in cc_emails)
            else:
                recipients.update(normalize_email(e) for e in re.split(r'[,;]', cc_emails) if e.strip())

            # Get thread info
            subject = email.get('subject', '')
            thread_id = extract_thread_id(subject)
            message_id = email.get('message_id', '')
            in_reply_to = email.get('in_reply_to', '')
            references = email.get('references', [])
            if isinstance(references, str):
                references = [r.strip() for r in references.split() if r.strip()]

            record = _EmailRecord(
                message_id=message_id,
                sender=sender,
                recipients=recipients,
                timestamp=email_date,
                thread_id=thread_id,
                in_reply_to=in_reply_to,
                references=references,
            )

            self._emails.append(record)

            if message_id:
                self._by_message_id[message_id] = record

            if thread_id:
                self._by_thread_id[thread_id].append(record)

            # Track emails received by each recipient from sender
            for recipient in recipients:
                self._received[(recipient, sender)].append(email_date)

        # Sort thread emails by timestamp
        for thread_id in self._by_thread_id:
            self._by_thread_id[thread_id].sort(key=lambda r: r.timestamp)

    def _detect_responses(self) -> None:
        """Detect response events from indexed emails."""
        seen_responses = set()  # (responder, original_sender, response_time) to dedupe

        # Method 1: Use in_reply_to field
        for record in self._emails:
            if record.in_reply_to and record.in_reply_to in self._by_message_id:
                original = self._by_message_id[record.in_reply_to]
                if original.sender != record.sender:  # Actual response, not self-reply
                    response_time_hours = (record.timestamp - original.timestamp).total_seconds() / 3600.0
                    if 0 < response_time_hours < 720:  # Max 30 days
                        key = (record.sender, original.sender, record.timestamp.isoformat())
                        if key not in seen_responses:
                            seen_responses.add(key)
                            self._responses[(record.sender, original.sender)].append(
                                (original.timestamp, record.timestamp)
                            )

        # Method 2: Thread-based detection via subject matching
        for thread_id, thread_emails in self._by_thread_id.items():
            if len(thread_emails) < 2:
                continue

            # For each email in thread, find the most recent email from different sender
            for i in range(1, len(thread_emails)):
                current = thread_emails[i]

                # Look backwards for the most recent email from a different sender
                for j in range(i - 1, -1, -1):
                    prev = thread_emails[j]
                    if prev.sender != current.sender and current.sender in prev.recipients:
                        # Found response: current is responding to prev
                        response_time_hours = (current.timestamp - prev.timestamp).total_seconds() / 3600.0
                        if 0 < response_time_hours < 720:
                            key = (current.sender, prev.sender, current.timestamp.isoformat())
                            if key not in seen_responses:
                                seen_responses.add(key)
                                self._responses[(current.sender, prev.sender)].append(
                                    (prev.timestamp, current.timestamp)
                                )
                        break  # Only count most recent prior email as the one being responded to

    def get_response_history(
        self,
        user: str,
        sender: str,
        reference_time: Optional[datetime] = None,
    ) -> UserResponse:
        """Get user response history for a specific sender.

        Args:
            user: User email address (the responder)
            sender: Sender email address (who the user responds to)
            reference_time: Time to compute history from (default: now)

        Returns:
            UserResponse with rate and timing metrics
        """
        if reference_time is None:
            reference_time = datetime.now()

        if reference_time.tzinfo is not None:
            reference_time = reference_time.replace(tzinfo=None)

        user = normalize_email(user)
        sender = normalize_email(sender)

        # Count emails from sender to user before reference time
        received_times = self._received.get((user, sender), [])
        total_received = sum(1 for t in received_times if t < reference_time)

        # Count responses from user to sender before reference time
        response_events = self._responses.get((user, sender), [])
        valid_responses = [
            (orig, resp) for orig, resp in response_events
            if resp < reference_time
        ]

        total_responses = len(valid_responses)
        response_times = [
            (resp - orig).total_seconds() / 3600.0
            for orig, resp in valid_responses
        ]

        # Compute metrics
        reply_rate = total_responses / total_received if total_received > 0 else 0.0
        avg_time = sum(response_times) / len(response_times) if response_times else None

        return UserResponse(
            user=user,
            sender=sender,
            reference_time=reference_time,
            user_replied_to_sender_rate=min(1.0, reply_rate),  # Cap at 1.0
            avg_response_time_hours=avg_time,
            total_emails_from_sender=total_received,
            total_responses_to_sender=total_responses,
            response_times=response_times,
        )

    def get_all_user_sender_pairs(self) -> list[tuple[str, str]]:
        """Get all (user, sender) pairs with email history."""
        pairs = set()
        for user, sender in self._received.keys():
            pairs.add((user, sender))
        return list(pairs)


def compute_user_response_from_emails(
    emails: list[dict],
    user: str,
    sender: str,
    reference_time: Optional[datetime] = None,
) -> UserResponse:
    """Compute user response history from a list of emails.

    This is the simple interface for one-off computations. For repeated
    lookups, use UserResponseIndex instead.

    Args:
        emails: List of email dictionaries
        user: User email address
        sender: Sender email address
        reference_time: Time to compute history from (default: now)

    Returns:
        UserResponse with rate and timing metrics
    """
    index = UserResponseIndex(emails)
    return index.get_response_history(user, sender, reference_time)


async def compute_user_response_from_db(
    db,
    user: str,
    sender: str,
    reference_time: Optional[datetime] = None,
) -> UserResponse:
    """Compute user response history using SurrealDB.

    Args:
        db: AsyncSurreal database connection
        user: User email address
        sender: Sender email address
        reference_time: Time to compute history from (default: now)

    Returns:
        UserResponse with metrics from database
    """
    if reference_time is None:
        reference_time = datetime.now()

    user = normalize_email(user)
    sender = normalize_email(sender)

    # Use the SurrealDB function
    result = await db.query(
        'RETURN fn::user_response_history($user, $sender, $reference_time)',
        {
            'user': user,
            'sender': sender,
            'reference_time': reference_time.isoformat(),
        }
    )

    # Parse result - new API returns list directly
    if result and isinstance(result, list) and len(result) > 0:
        data = result[0]
        return UserResponse(
            user=user,
            sender=sender,
            reference_time=reference_time,
            user_replied_to_sender_rate=data.get('reply_rate', 0.0) or 0.0,
            avg_response_time_hours=data.get('avg_response_time_hours'),
            total_emails_from_sender=data.get('total_received', 0) or 0,
            total_responses_to_sender=data.get('total_responses', 0) or 0,
        )

    return UserResponse(
        user=user,
        sender=sender,
        reference_time=reference_time,
    )


def compute_user_response_from_db_sync(
    db,
    user: str,
    sender: str,
    reference_time: Optional[datetime] = None,
) -> UserResponse:
    """Synchronous version of compute_user_response_from_db.

    Args:
        db: Surreal database connection (sync client)
        user: User email address
        sender: Sender email address
        reference_time: Time to compute history from (default: now)

    Returns:
        UserResponse with metrics from database
    """
    if reference_time is None:
        reference_time = datetime.now()

    user = normalize_email(user)
    sender = normalize_email(sender)

    result = db.query(
        'RETURN fn::user_response_history($user, $sender, $reference_time)',
        {
            'user': user,
            'sender': sender,
            'reference_time': reference_time.isoformat(),
        }
    )

    if result and isinstance(result, list) and len(result) > 0:
        data = result[0]
        return UserResponse(
            user=user,
            sender=sender,
            reference_time=reference_time,
            user_replied_to_sender_rate=data.get('reply_rate', 0.0) or 0.0,
            avg_response_time_hours=data.get('avg_response_time_hours'),
            total_emails_from_sender=data.get('total_received', 0) or 0,
            total_responses_to_sender=data.get('total_responses', 0) or 0,
        )

    return UserResponse(
        user=user,
        sender=sender,
        reference_time=reference_time,
    )


if __name__ == '__main__':
    # Example usage
    from datetime import datetime, timedelta

    # Sample email thread
    sample_emails = [
        {
            'message_id': 'msg1@example.com',
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Project update',
            'date': (datetime.now() - timedelta(days=5, hours=10)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'message_id': 'msg2@example.com',
            'from': 'bob@example.com',
            'to': 'alice@example.com',
            'subject': 'Re: Project update',
            'in_reply_to': 'msg1@example.com',
            'date': (datetime.now() - timedelta(days=5, hours=8)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'message_id': 'msg3@example.com',
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'New task',
            'date': (datetime.now() - timedelta(days=3)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'message_id': 'msg4@example.com',
            'from': 'bob@example.com',
            'to': 'alice@example.com',
            'subject': 'Re: New task',
            'in_reply_to': 'msg3@example.com',
            'date': (datetime.now() - timedelta(days=2, hours=20)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'message_id': 'msg5@example.com',
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'subject': 'Quick question',
            'date': (datetime.now() - timedelta(days=1)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        # No response to msg5 - tests that not all emails are responded to
    ]

    print("=" * 60)
    print("USER RESPONSE HISTORY COMPUTATION")
    print("=" * 60)
    print()

    # Build index
    index = UserResponseIndex(sample_emails)

    # Bob's response history to Alice
    response = index.get_response_history('bob@example.com', 'alice@example.com')

    print("Bob's response history to Alice:")
    print(f"  Emails received from Alice: {response.total_emails_from_sender}")
    print(f"  Responses sent to Alice:    {response.total_responses_to_sender}")
    print(f"  Reply rate:                 {response.user_replied_to_sender_rate:.2%}")
    if response.avg_response_time_hours:
        print(f"  Avg response time:          {response.avg_response_time_hours:.1f} hours")
    if response.median_response_time_hours:
        print(f"  Median response time:       {response.median_response_time_hours:.1f} hours")
    print()

    # Alice's response history to Bob
    response_alice = index.get_response_history('alice@example.com', 'bob@example.com')

    print("Alice's response history to Bob:")
    print(f"  Emails received from Bob:   {response_alice.total_emails_from_sender}")
    print(f"  Responses sent to Bob:      {response_alice.total_responses_to_sender}")
    print(f"  Reply rate:                 {response_alice.user_replied_to_sender_rate:.2%}")
    if response_alice.avg_response_time_hours:
        print(f"  Avg response time:          {response_alice.avg_response_time_hours:.1f} hours")
    print()

    print("Feature vector (normalized):")
    print(f"  {response.to_feature_vector()}")
    print()
    print("=" * 60)
