#!/usr/bin/env python3
"""Time-windowed email frequency computation.

Computes email frequency from each sender in configurable time windows (7d, 30d, 90d).
Supports both SurrealDB queries and in-memory computation from email lists.

Output features:
- emails_from_sender_7d: Count of emails from sender in last 7 days
- emails_from_sender_30d: Count of emails from sender in last 30 days
- emails_from_sender_90d: Count of emails from sender in last 90 days
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional


# Default time windows in days
DEFAULT_WINDOWS = (7, 30, 90)


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


@dataclass
class SenderFrequency:
    """Time-windowed email frequency from a sender.

    Attributes:
        sender: Normalized sender email address
        recipient: Normalized recipient email address
        reference_time: Time from which windows are calculated
        emails_from_sender_7d: Emails from sender in last 7 days
        emails_from_sender_30d: Emails from sender in last 30 days
        emails_from_sender_90d: Emails from sender in last 90 days
    """
    sender: str
    recipient: str
    reference_time: datetime
    emails_from_sender_7d: int = 0
    emails_from_sender_30d: int = 0
    emails_from_sender_90d: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'reference_time': self.reference_time.isoformat(),
            'emails_from_sender_7d': self.emails_from_sender_7d,
            'emails_from_sender_30d': self.emails_from_sender_30d,
            'emails_from_sender_90d': self.emails_from_sender_90d,
        }

    def to_feature_vector(self) -> list[float]:
        """Convert to feature vector for ML pipeline.

        Returns 3-dimensional normalized vector.
        """
        return [
            min(self.emails_from_sender_7d, 50) / 50.0,   # Cap at 50 (~7/day)
            min(self.emails_from_sender_30d, 100) / 100.0,  # Cap at 100
            min(self.emails_from_sender_90d, 200) / 200.0,  # Cap at 200
        ]


def compute_sender_frequency_from_emails(
    emails: list[dict],
    sender: str,
    recipient: str,
    reference_time: Optional[datetime] = None,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> SenderFrequency:
    """Compute time-windowed email frequency from a list of emails.

    Args:
        emails: List of email dictionaries with 'from', 'to', 'date' fields
        sender: Sender email address to count
        recipient: Recipient email address (user)
        reference_time: Time to compute windows from (default: now)
        windows: Tuple of window sizes in days (default: 7, 30, 90)

    Returns:
        SenderFrequency with counts for each window
    """
    if reference_time is None:
        reference_time = datetime.now()

    # Make reference_time timezone-naive for comparison if needed
    if reference_time.tzinfo is not None:
        reference_time_naive = reference_time.replace(tzinfo=None)
    else:
        reference_time_naive = reference_time

    sender = normalize_email(sender)
    recipient = normalize_email(recipient)

    # Compute window boundaries
    window_starts = {
        days: reference_time_naive - timedelta(days=days)
        for days in windows
    }

    # Count emails in each window
    counts = {days: 0 for days in windows}

    for email in emails:
        email_sender = normalize_email(email.get('from', ''))
        if email_sender != sender:
            continue

        # Check if recipient is in to/cc fields
        to_emails = email.get('to', '') or ''
        cc_emails = email.get('cc', '') or ''

        # Handle both string and list formats
        if isinstance(to_emails, list):
            all_recipients = [normalize_email(e) for e in to_emails]
        else:
            all_recipients = [normalize_email(e) for e in re.split(r'[,;]', to_emails)]

        if isinstance(cc_emails, list):
            all_recipients.extend([normalize_email(e) for e in cc_emails])
        else:
            all_recipients.extend([normalize_email(e) for e in re.split(r'[,;]', cc_emails)])

        if recipient not in all_recipients:
            continue

        # Parse email date
        email_date = parse_date(email.get('date', ''))
        if email_date is None:
            continue

        # Make email_date timezone-naive for comparison
        if email_date.tzinfo is not None:
            email_date = email_date.replace(tzinfo=None)

        # Check if email is before reference time
        if email_date >= reference_time_naive:
            continue

        # Count for each window
        for days in windows:
            if email_date >= window_starts[days]:
                counts[days] += 1

    return SenderFrequency(
        sender=sender,
        recipient=recipient,
        reference_time=reference_time,
        emails_from_sender_7d=counts.get(7, 0),
        emails_from_sender_30d=counts.get(30, 0),
        emails_from_sender_90d=counts.get(90, 0),
    )


async def compute_sender_frequency_from_db(
    db,
    sender: str,
    recipient: str,
    reference_time: Optional[datetime] = None,
) -> SenderFrequency:
    """Compute time-windowed email frequency using SurrealDB.

    Args:
        db: AsyncSurreal database connection
        sender: Sender email address
        recipient: Recipient email address
        reference_time: Time to compute windows from (default: now)

    Returns:
        SenderFrequency with counts from database
    """
    if reference_time is None:
        reference_time = datetime.now()

    sender = normalize_email(sender)
    recipient = normalize_email(recipient)

    # Use the SurrealDB function
    result = await db.query(
        'RETURN fn::sender_frequency_all_windows($sender, $recipient, $reference_time)',
        {
            'sender': sender,
            'recipient': recipient,
            'reference_time': reference_time.isoformat(),
        }
    )

    # Parse result - new API returns list directly
    if result and isinstance(result, list) and len(result) > 0:
        data = result[0]
        return SenderFrequency(
            sender=sender,
            recipient=recipient,
            reference_time=reference_time,
            emails_from_sender_7d=data.get('emails_7d', 0) or 0,
            emails_from_sender_30d=data.get('emails_30d', 0) or 0,
            emails_from_sender_90d=data.get('emails_90d', 0) or 0,
        )

    return SenderFrequency(
        sender=sender,
        recipient=recipient,
        reference_time=reference_time,
    )


def compute_sender_frequency_from_db_sync(
    db,
    sender: str,
    recipient: str,
    reference_time: Optional[datetime] = None,
) -> SenderFrequency:
    """Synchronous version of compute_sender_frequency_from_db.

    Args:
        db: Surreal database connection (sync client)
        sender: Sender email address
        recipient: Recipient email address
        reference_time: Time to compute windows from (default: now)

    Returns:
        SenderFrequency with counts from database
    """
    if reference_time is None:
        reference_time = datetime.now()

    sender = normalize_email(sender)
    recipient = normalize_email(recipient)

    result = db.query(
        'RETURN fn::sender_frequency_all_windows($sender, $recipient, $reference_time)',
        {
            'sender': sender,
            'recipient': recipient,
            'reference_time': reference_time.isoformat(),
        }
    )

    if result and isinstance(result, list) and len(result) > 0:
        data = result[0]
        return SenderFrequency(
            sender=sender,
            recipient=recipient,
            reference_time=reference_time,
            emails_from_sender_7d=data.get('emails_7d', 0) or 0,
            emails_from_sender_30d=data.get('emails_30d', 0) or 0,
            emails_from_sender_90d=data.get('emails_90d', 0) or 0,
        )

    return SenderFrequency(
        sender=sender,
        recipient=recipient,
        reference_time=reference_time,
    )


class SenderFrequencyIndex:
    """Index for efficient time-windowed frequency lookups from email lists.

    Pre-indexes emails by sender for fast repeated lookups.
    """

    def __init__(self, emails: list[dict]):
        """Build index from email list.

        Args:
            emails: List of email dictionaries
        """
        # Index: sender -> list of (date, recipients)
        self._sender_index: dict[str, list[tuple[datetime, set[str]]]] = defaultdict(list)

        for email in emails:
            sender = normalize_email(email.get('from', ''))
            if not sender:
                continue

            email_date = parse_date(email.get('date', ''))
            if email_date is None:
                continue

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

            # Make date timezone-naive for consistent comparison
            if email_date.tzinfo is not None:
                email_date = email_date.replace(tzinfo=None)

            self._sender_index[sender].append((email_date, recipients))

        # Sort each sender's emails by date for efficient windowed queries
        for sender in self._sender_index:
            self._sender_index[sender].sort(key=lambda x: x[0])

    def get_frequency(
        self,
        sender: str,
        recipient: str,
        reference_time: Optional[datetime] = None,
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
    ) -> SenderFrequency:
        """Get time-windowed frequency for a sender-recipient pair.

        Args:
            sender: Sender email address
            recipient: Recipient email address
            reference_time: Time to compute windows from (default: now)
            windows: Tuple of window sizes in days

        Returns:
            SenderFrequency with counts
        """
        if reference_time is None:
            reference_time = datetime.now()

        if reference_time.tzinfo is not None:
            reference_time = reference_time.replace(tzinfo=None)

        sender = normalize_email(sender)
        recipient = normalize_email(recipient)

        # Get sender's emails
        sender_emails = self._sender_index.get(sender, [])
        if not sender_emails:
            return SenderFrequency(
                sender=sender,
                recipient=recipient,
                reference_time=reference_time,
            )

        # Compute window boundaries
        window_starts = {
            days: reference_time - timedelta(days=days)
            for days in windows
        }

        counts = {days: 0 for days in windows}

        for email_date, recipients in sender_emails:
            # Skip emails at or after reference time
            if email_date >= reference_time:
                continue

            # Check if recipient is in this email
            if recipient not in recipients:
                continue

            # Count for applicable windows
            for days in windows:
                if email_date >= window_starts[days]:
                    counts[days] += 1

        return SenderFrequency(
            sender=sender,
            recipient=recipient,
            reference_time=reference_time,
            emails_from_sender_7d=counts.get(7, 0),
            emails_from_sender_30d=counts.get(30, 0),
            emails_from_sender_90d=counts.get(90, 0),
        )

    def get_all_senders(self) -> list[str]:
        """Get list of all indexed senders."""
        return list(self._sender_index.keys())

    def get_sender_total(self, sender: str) -> int:
        """Get total email count for a sender."""
        return len(self._sender_index.get(normalize_email(sender), []))


if __name__ == '__main__':
    # Example usage
    from datetime import datetime

    # Sample emails
    sample_emails = [
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'date': (datetime.now() - timedelta(days=2)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'date': (datetime.now() - timedelta(days=5)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'date': (datetime.now() - timedelta(days=15)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'date': (datetime.now() - timedelta(days=45)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'from': 'alice@example.com',
            'to': 'bob@example.com',
            'date': (datetime.now() - timedelta(days=60)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
        {
            'from': 'charlie@example.com',
            'to': 'bob@example.com',
            'date': (datetime.now() - timedelta(days=3)).strftime('%a, %d %b %Y %H:%M:%S +0000'),
        },
    ]

    print("=" * 60)
    print("SENDER FREQUENCY COMPUTATION")
    print("=" * 60)
    print()

    # Method 1: Direct computation
    freq = compute_sender_frequency_from_emails(
        sample_emails,
        sender='alice@example.com',
        recipient='bob@example.com',
    )

    print("Direct computation for alice@example.com -> bob@example.com:")
    print(f"  7d:  {freq.emails_from_sender_7d}")
    print(f"  30d: {freq.emails_from_sender_30d}")
    print(f"  90d: {freq.emails_from_sender_90d}")
    print()

    # Method 2: Using index (efficient for multiple lookups)
    index = SenderFrequencyIndex(sample_emails)

    print("Indexed computation for alice@example.com -> bob@example.com:")
    freq_indexed = index.get_frequency('alice@example.com', 'bob@example.com')
    print(f"  7d:  {freq_indexed.emails_from_sender_7d}")
    print(f"  30d: {freq_indexed.emails_from_sender_30d}")
    print(f"  90d: {freq_indexed.emails_from_sender_90d}")
    print()

    print("Indexed computation for charlie@example.com -> bob@example.com:")
    freq_charlie = index.get_frequency('charlie@example.com', 'bob@example.com')
    print(f"  7d:  {freq_charlie.emails_from_sender_7d}")
    print(f"  30d: {freq_charlie.emails_from_sender_30d}")
    print(f"  90d: {freq_charlie.emails_from_sender_90d}")
    print()

    print("Feature vector (normalized):")
    print(f"  {freq.to_feature_vector()}")
    print()
    print("=" * 60)
