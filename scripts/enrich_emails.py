#!/usr/bin/env python3
"""Stage 5: Enrich Gmail emails with action labels.

Computes action labels from behavioral signals:
- Priority from labels (STARRED, IMPORTANT, SPAM, etc.)
- Action from thread structure (REPLIED, FORWARDED, IGNORED)
- Response timing for replied emails

Output: enriched_emails.jsonl with action/timing/priority fields
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# Priority mappings from Gmail labels
PRIORITY_LABELS = {
    'STARRED': 'high',
    'IMPORTANT': 'medium',
    'CATEGORY PROMOTIONS': 'low',
    'SPAM': 'lowest',
}

# Response time thresholds in seconds
RESPONSE_TIME_THRESHOLDS = {
    'IMMEDIATE': 3600,        # < 1 hour
    'SAME_DAY': 86400,        # < 24 hours
    'NEXT_DAY': 172800,       # < 48 hours
    # LATER: >= 48 hours
}

# Age threshold for IGNORED classification (days)
IGNORED_AGE_DAYS = 7


def parse_iso_date(date_str: str) -> Optional[datetime]:
    """Parse ISO format date string."""
    if not date_str:
        return None
    try:
        # Handle ISO format with Z suffix
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


def compute_priority(labels: list[str]) -> str:
    """Compute priority from Gmail labels.

    Priority hierarchy (highest wins):
    1. STARRED → high
    2. IMPORTANT → medium
    3. CATEGORY PROMOTIONS → low
    4. SPAM → lowest
    5. Default → normal
    """
    for label, priority in PRIORITY_LABELS.items():
        if label in labels:
            return priority
    return 'normal'


def classify_response_time(seconds: float) -> str:
    """Classify response time into buckets."""
    if seconds < RESPONSE_TIME_THRESHOLDS['IMMEDIATE']:
        return 'IMMEDIATE'
    elif seconds < RESPONSE_TIME_THRESHOLDS['SAME_DAY']:
        return 'SAME_DAY'
    elif seconds < RESPONSE_TIME_THRESHOLDS['NEXT_DAY']:
        return 'NEXT_DAY'
    else:
        return 'LATER'


def is_reply_subject(subject: str) -> bool:
    """Check if subject indicates a reply."""
    subject_lower = subject.lower().strip()
    return subject_lower.startswith(('re:', 're: '))


def is_forward_subject(subject: str) -> bool:
    """Check if subject indicates a forward."""
    subject_lower = subject.lower().strip()
    return subject_lower.startswith(('fw:', 'fwd:', 'forwarded:'))


def build_thread_index(emails: list[dict]) -> dict[str, list[dict]]:
    """Group emails by thread_id and sort by date."""
    threads = defaultdict(list)
    for email in emails:
        thread_id = email.get('thread_id', '')
        if thread_id:
            threads[thread_id].append(email)

    # Sort each thread by date
    for thread_id in threads:
        threads[thread_id].sort(
            key=lambda e: e.get('date', '') or ''
        )

    return dict(threads)


def make_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert datetime to naive (no timezone)."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def compute_action_for_received(
    email: dict,
    thread_emails: list[dict],
    my_email: str,
    reference_date: datetime
) -> tuple[str, Optional[str], Optional[float]]:
    """Compute action label for a received email.

    Args:
        email: The received email to analyze
        thread_emails: All emails in the same thread (sorted by date)
        my_email: The user's email address
        reference_date: Current date for age calculations

    Returns:
        Tuple of (action, response_timing, response_seconds)
    """
    email_date = make_naive(parse_iso_date(email.get('date', '')))
    reference_date = make_naive(reference_date)
    email_idx = None
    labels = email.get('labels', [])

    # Find this email's position in thread
    for i, e in enumerate(thread_emails):
        if e.get('message_id') == email.get('message_id'):
            email_idx = i
            break

    if email_idx is None:
        return 'UNKNOWN', None, None

    # Look for subsequent sent emails that are replies/forwards
    for later_email in thread_emails[email_idx + 1:]:
        later_labels = later_email.get('labels', [])
        if 'SENT' not in later_labels:
            continue

        # Check if from me
        from_field = later_email.get('from', '').lower()
        if my_email.lower() not in from_field:
            continue

        later_subject = later_email.get('subject', '')
        later_date = make_naive(parse_iso_date(later_email.get('date', '')))

        if is_reply_subject(later_subject):
            # Compute response time
            if email_date and later_date:
                delta = (later_date - email_date).total_seconds()
                if delta >= 0:
                    timing = classify_response_time(delta)
                    return 'REPLIED', timing, delta
            return 'REPLIED', 'UNKNOWN', None

        if is_forward_subject(later_subject):
            return 'FORWARDED', None, None

    # No reply/forward found - check for IGNORED
    if 'UNREAD' in labels:
        if email_date and reference_date:
            age_days = (reference_date - email_date).days
            if age_days > IGNORED_AGE_DAYS:
                return 'IGNORED', 'NEVER', None
        return 'READ_PENDING', 'NEVER', None

    # Was opened but not replied to
    if 'OPENED' in labels or 'UNREAD' not in labels:
        if email_date and reference_date:
            age_days = (reference_date - email_date).days
            if age_days > IGNORED_AGE_DAYS:
                return 'ARCHIVED', 'NEVER', None

    return 'UNKNOWN', None, None


def enrich_emails(
    input_path: Path,
    output_path: Path,
    my_email: str,
    stats_path: Optional[Path] = None
) -> dict:
    """Enrich emails with action labels.

    Args:
        input_path: Path to cleaned_emails.jsonl
        output_path: Path for enriched_emails.jsonl
        my_email: User's email address for identifying sent emails
        stats_path: Optional path for enrichment_stats.json

    Returns:
        Statistics dictionary
    """
    print(f"Loading emails from {input_path}...")

    emails = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                emails.append(json.loads(line))

    print(f"Loaded {len(emails)} emails")

    # Build thread index
    print("Building thread index...")
    threads = build_thread_index(emails)
    print(f"Found {len(threads)} threads")

    # Find reference date (latest email date)
    reference_date = datetime.now()
    for email in emails:
        email_date = parse_iso_date(email.get('date', ''))
        if email_date:
            # Use latest email date as reference
            if email_date.replace(tzinfo=None) > reference_date.replace(tzinfo=None) if reference_date.tzinfo else reference_date:
                reference_date = email_date.replace(tzinfo=None) if email_date.tzinfo else email_date

    # Use the max date from emails as reference
    max_date = None
    for email in emails:
        d = parse_iso_date(email.get('date', ''))
        if d:
            d_naive = d.replace(tzinfo=None) if d.tzinfo else d
            if max_date is None or d_naive > max_date:
                max_date = d_naive
    if max_date:
        reference_date = max_date

    print(f"Reference date: {reference_date}")

    # Statistics
    stats = {
        'total_emails': len(emails),
        'sent_emails': 0,
        'received_emails': 0,
        'priority_distribution': Counter(),
        'action_distribution': Counter(),
        'response_timing_distribution': Counter(),
        'response_time_stats': {
            'count': 0,
            'min_seconds': None,
            'max_seconds': None,
            'total_seconds': 0,
        }
    }

    # Enrich emails
    print("Enriching emails...")
    enriched = []

    for email in emails:
        labels = email.get('labels', [])
        thread_id = email.get('thread_id', '')
        thread_emails = threads.get(thread_id, [email])

        # Compute priority
        priority = compute_priority(labels)
        email['priority'] = priority
        stats['priority_distribution'][priority] += 1

        # Determine if sent or received
        is_sent = 'SENT' in labels

        if is_sent:
            stats['sent_emails'] += 1
            email['action'] = 'COMPOSED'  # Sent emails are compositions
            email['response_timing'] = None
            email['response_time_seconds'] = None
        else:
            stats['received_emails'] += 1

            # Compute action for received emails
            action, timing, response_seconds = compute_action_for_received(
                email, thread_emails, my_email, reference_date
            )

            email['action'] = action
            email['response_timing'] = timing
            email['response_time_seconds'] = response_seconds

            stats['action_distribution'][action] += 1

            if timing:
                stats['response_timing_distribution'][timing] += 1

            if response_seconds is not None:
                rs = stats['response_time_stats']
                rs['count'] += 1
                rs['total_seconds'] += response_seconds
                if rs['min_seconds'] is None or response_seconds < rs['min_seconds']:
                    rs['min_seconds'] = response_seconds
                if rs['max_seconds'] is None or response_seconds > rs['max_seconds']:
                    rs['max_seconds'] = response_seconds

        enriched.append(email)

    # Compute average response time
    rs = stats['response_time_stats']
    if rs['count'] > 0:
        rs['avg_seconds'] = rs['total_seconds'] / rs['count']
        rs['avg_hours'] = rs['avg_seconds'] / 3600
    else:
        rs['avg_seconds'] = None
        rs['avg_hours'] = None

    # Convert Counters to dicts for JSON
    stats['priority_distribution'] = dict(stats['priority_distribution'])
    stats['action_distribution'] = dict(stats['action_distribution'])
    stats['response_timing_distribution'] = dict(stats['response_timing_distribution'])

    # Write enriched emails
    print(f"Writing {len(enriched)} enriched emails to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for email in enriched:
            f.write(json.dumps(email, ensure_ascii=False) + '\n')

    # Write stats
    if stats_path:
        print(f"Writing stats to {stats_path}...")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def print_stats(stats: dict) -> None:
    """Print enrichment statistics."""
    print()
    print("=" * 60)
    print("ENRICHMENT STATISTICS")
    print("=" * 60)
    print(f"Total emails: {stats['total_emails']}")
    print(f"  Sent: {stats['sent_emails']}")
    print(f"  Received: {stats['received_emails']}")
    print()

    print("Priority distribution:")
    for priority in ['high', 'medium', 'normal', 'low', 'lowest']:
        count = stats['priority_distribution'].get(priority, 0)
        pct = 100 * count / stats['total_emails'] if stats['total_emails'] else 0
        print(f"  {priority:10s}: {count:7d} ({pct:5.1f}%)")
    print()

    print("Action distribution (received emails):")
    for action, count in sorted(stats['action_distribution'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['received_emails'] if stats['received_emails'] else 0
        print(f"  {action:15s}: {count:7d} ({pct:5.1f}%)")
    print()

    print("Response timing (for replied emails):")
    for timing in ['IMMEDIATE', 'SAME_DAY', 'NEXT_DAY', 'LATER', 'NEVER', 'UNKNOWN']:
        count = stats['response_timing_distribution'].get(timing, 0)
        total_timed = sum(stats['response_timing_distribution'].values())
        pct = 100 * count / total_timed if total_timed else 0
        print(f"  {timing:10s}: {count:7d} ({pct:5.1f}%)")
    print()

    rs = stats['response_time_stats']
    print("Response time statistics:")
    print(f"  Emails with computed response time: {rs['count']}")
    if rs['count'] > 0:
        print(f"  Min: {rs['min_seconds']:.0f}s ({rs['min_seconds']/3600:.2f}h)")
        print(f"  Max: {rs['max_seconds']:.0f}s ({rs['max_seconds']/3600:.2f}h)")
        print(f"  Avg: {rs['avg_seconds']:.0f}s ({rs['avg_hours']:.2f}h)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 5: Enrich Gmail emails with action labels'
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Path to cleaned_emails.jsonl'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('enriched_emails.jsonl'),
        help='Output path for enriched emails (default: enriched_emails.jsonl)'
    )
    parser.add_argument(
        '-s', '--stats',
        type=Path,
        help='Output path for enrichment stats JSON'
    )
    parser.add_argument(
        '-e', '--email',
        type=str,
        required=True,
        help='Your email address (for identifying sent emails)'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    stats = enrich_emails(
        args.input,
        args.output,
        args.email,
        args.stats
    )

    print_stats(stats)


if __name__ == '__main__':
    main()
