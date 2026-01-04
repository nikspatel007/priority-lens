#!/usr/bin/env python3
"""Label user actions from Enron email dataset.

Maps email labels to 5-class action space for RL training:
- REPLY_NOW: User replied within 1 hour
- REPLY_LATER: User replied after 1 hour
- FORWARD: User forwarded the email
- ARCHIVE: User archived to folder (not inbox/sent/deleted)
- DELETE: User deleted the email

Response time is computed by matching sent replies to original messages
using In-Reply-To and References headers.

Labels dropped (not actions on incoming email):
- COMPOSED: New composition (not a response to incoming)
- KEPT: Left in inbox (no action taken)
- JUNK: Junk folder (system-handled, not user action)
- AUTO_FILED: System auto-filing
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import Counter


# 5-class action space
ACTION_CLASSES = ['REPLY_NOW', 'REPLY_LATER', 'FORWARD', 'ARCHIVE', 'DELETE']

# Response time threshold in seconds (1 hour)
REPLY_NOW_THRESHOLD_SECONDS = 3600

# Folder classifications
SENT_FOLDERS = {'sent', 'sent_items', '_sent_mail', '_sent'}
DELETED_FOLDERS = {'deleted_items', 'connect_deletes'}
INBOX_FOLDERS = {'inbox', 'notes_inbox'}
JUNK_FOLDERS = {'junk', 'junk_file'}
AUTO_FILED_FOLDERS = {'all_documents', 'discussion_threads'}


def classify_folder(folder: str) -> str:
    """Classify a folder into a category.

    Returns:
        One of: 'sent', 'deleted', 'inbox', 'junk', 'auto_filed', 'archived'
    """
    folder_lower = folder.lower().strip()

    if folder_lower in SENT_FOLDERS:
        return 'sent'
    elif folder_lower in DELETED_FOLDERS:
        return 'deleted'
    elif folder_lower in INBOX_FOLDERS:
        return 'inbox'
    elif folder_lower in JUNK_FOLDERS:
        return 'junk'
    elif folder_lower in AUTO_FILED_FOLDERS:
        return 'auto_filed'
    else:
        return 'archived'


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse email date string to datetime.

    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S",
        "%a %b %d %H:%M:%S %Y",
    ]

    # Clean timezone abbreviations
    cleaned = date_str.replace(" (PDT)", "").replace(" (PST)", "")
    cleaned = cleaned.replace(" (EDT)", "").replace(" (EST)", "")
    cleaned = cleaned.replace(" (CDT)", "").replace(" (CST)", "")
    cleaned = cleaned.replace(" (MDT)", "").replace(" (MST)", "")

    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    return None


def extract_message_id(msg_id: str) -> str:
    """Extract clean message ID from header.

    Handles formats like: <ABC123@example.com> or ABC123@example.com
    """
    if not msg_id:
        return ''
    msg_id = msg_id.strip()
    if msg_id.startswith('<') and msg_id.endswith('>'):
        return msg_id[1:-1]
    return msg_id


def build_message_index(emails: list[dict]) -> dict[str, dict]:
    """Build index of emails by message_id for response time lookups.

    Args:
        emails: List of email dictionaries

    Returns:
        Dict mapping message_id -> email dict
    """
    index = {}
    for email in emails:
        msg_id = extract_message_id(email.get('message_id', ''))
        if msg_id:
            index[msg_id] = email
    return index


def compute_response_time(
    reply_email: dict,
    message_index: dict[str, dict]
) -> Optional[float]:
    """Compute response time for a reply email.

    Looks up the original message using In-Reply-To or References headers
    and computes the time delta in seconds.

    Args:
        reply_email: The reply email dict
        message_index: Index of message_id -> email

    Returns:
        Response time in seconds, or None if original not found
    """
    # Try In-Reply-To first
    in_reply_to = extract_message_id(reply_email.get('in_reply_to', ''))
    original = message_index.get(in_reply_to)

    # Try References if In-Reply-To didn't work
    if original is None:
        references = reply_email.get('references', '')
        if references:
            # References lists message IDs from oldest to newest, last is direct parent
            ref_ids = [extract_message_id(r) for r in references.split()]
            for ref_id in reversed(ref_ids):
                original = message_index.get(ref_id)
                if original:
                    break

    if original is None:
        return None

    # Parse dates
    reply_date = parse_date(reply_email.get('date', ''))
    original_date = parse_date(original.get('date', ''))

    if reply_date is None or original_date is None:
        return None

    # Compute delta (make timezone-naive for comparison if needed)
    try:
        if reply_date.tzinfo and not original_date.tzinfo:
            reply_date = reply_date.replace(tzinfo=None)
        elif original_date.tzinfo and not reply_date.tzinfo:
            original_date = original_date.replace(tzinfo=None)

        delta = reply_date - original_date
        return delta.total_seconds()
    except (TypeError, ValueError):
        return None


def infer_action(email: dict, message_index: Optional[dict] = None) -> Optional[str]:
    """Infer the user action for an email in 5-class action space.

    Args:
        email: Email dictionary with folder, subject, etc.
        message_index: Optional message_id index for response time computation

    Returns:
        Action label: REPLY_NOW, REPLY_LATER, FORWARD, ARCHIVE, DELETE
        Returns None for non-actionable emails (COMPOSED, KEPT, JUNK, AUTO_FILED)
    """
    folder = email.get('folder', '')
    subject = email.get('subject', '')
    folder_type = classify_folder(folder)

    if folder_type == 'sent':
        subject_lower = subject.lower().strip()

        if subject_lower.startswith('re:'):
            # This is a reply - determine response time bucket
            if message_index:
                response_time = compute_response_time(email, message_index)
                if response_time is not None and response_time >= 0:
                    if response_time < REPLY_NOW_THRESHOLD_SECONDS:
                        return 'REPLY_NOW'
                    else:
                        return 'REPLY_LATER'
            # Default to REPLY_LATER if we can't compute response time
            return 'REPLY_LATER'

        elif subject_lower.startswith(('fw:', 'fwd:', 'forwarded:')):
            return 'FORWARD'

        else:
            # New composition - not an action on incoming email
            return None

    elif folder_type == 'deleted':
        return 'DELETE'

    elif folder_type == 'inbox':
        # Still in inbox = no action taken
        return None

    elif folder_type == 'junk':
        # Junk filtering is system-handled, not user action
        return None

    elif folder_type == 'auto_filed':
        # System auto-filing is not user action
        return None

    else:
        # In a user-created folder = user archived
        return 'ARCHIVE'


def process_emails(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None
) -> dict:
    """Process emails and add action labels.

    Args:
        input_path: Path to input emails.json
        output_path: Path for output labeled JSON
        limit: Optional limit on number of emails

    Returns:
        Statistics dictionary
    """
    print(f"Loading emails from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        emails = json.load(f)

    print(f"Loaded {len(emails)} emails")

    if limit:
        emails = emails[:limit]
        print(f"Limited to {limit} emails")

    # Build message index for response time computation
    print("Building message index...")
    message_index = build_message_index(emails)
    print(f"Indexed {len(message_index)} message IDs")

    # Count actions and response times
    action_counts = Counter()
    folder_type_counts = Counter()
    response_time_stats = {'matched': 0, 'unmatched': 0, 'reply_now': 0, 'reply_later': 0}
    labeled_emails = []

    # Add labels
    for email in emails:
        action = infer_action(email, message_index)

        # Track folder types for all emails
        folder_type = classify_folder(email.get('folder', ''))
        folder_type_counts[folder_type] += 1

        if action is None:
            # Skip non-actionable emails
            continue

        email['action'] = action
        action_counts[action] += 1
        labeled_emails.append(email)

        # Track response time matching for replies
        if action in ('REPLY_NOW', 'REPLY_LATER'):
            response_time = compute_response_time(email, message_index)
            if response_time is not None:
                response_time_stats['matched'] += 1
                if action == 'REPLY_NOW':
                    response_time_stats['reply_now'] += 1
                else:
                    response_time_stats['reply_later'] += 1
            else:
                response_time_stats['unmatched'] += 1

    # Write output
    print(f"Writing {len(labeled_emails)} labeled emails to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_emails, f, indent=2, ensure_ascii=False)

    stats = {
        'total_input': len(emails),
        'total_labeled': len(labeled_emails),
        'dropped': len(emails) - len(labeled_emails),
        'action_counts': dict(action_counts),
        'folder_type_counts': dict(folder_type_counts),
        'response_time_stats': response_time_stats
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Label user actions in Enron email dataset (5-class action space)'
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Path to input emails.json'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data/emails_labeled.json'),
        help='Output path for labeled emails (default: data/emails_labeled.json)'
    )
    parser.add_argument(
        '-n', '--limit',
        type=int,
        help='Limit number of emails to process'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only compute and print statistics, do not write output'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    if args.stats_only:
        # Just compute stats without writing
        print(f"Computing statistics for {args.input}...")
        with open(args.input, 'r', encoding='utf-8') as f:
            emails = json.load(f)

        message_index = build_message_index(emails)
        action_counts = Counter()
        folder_type_counts = Counter()
        response_time_stats = {'matched': 0, 'unmatched': 0, 'reply_now': 0, 'reply_later': 0}
        dropped = 0

        for email in emails:
            action = infer_action(email, message_index)
            folder_type = classify_folder(email.get('folder', ''))
            folder_type_counts[folder_type] += 1

            if action is None:
                dropped += 1
                continue

            action_counts[action] += 1

            if action in ('REPLY_NOW', 'REPLY_LATER'):
                response_time = compute_response_time(email, message_index)
                if response_time is not None:
                    response_time_stats['matched'] += 1
                    if action == 'REPLY_NOW':
                        response_time_stats['reply_now'] += 1
                    else:
                        response_time_stats['reply_later'] += 1
                else:
                    response_time_stats['unmatched'] += 1

        stats = {
            'total_input': len(emails),
            'total_labeled': len(emails) - dropped,
            'dropped': dropped,
            'action_counts': dict(action_counts),
            'folder_type_counts': dict(folder_type_counts),
            'response_time_stats': response_time_stats
        }
    else:
        stats = process_emails(args.input, args.output, args.limit)

    # Print statistics
    print()
    print("=" * 60)
    print("LABELING STATISTICS (5-class action space)")
    print("=" * 60)
    print(f"Total input emails: {stats['total_input']}")
    print(f"Labeled (actionable): {stats['total_labeled']}")
    print(f"Dropped (non-actionable): {stats['dropped']}")
    print()
    print("Action distribution:")
    for action in ACTION_CLASSES:
        count = stats['action_counts'].get(action, 0)
        if stats['total_labeled'] > 0:
            pct = 100 * count / stats['total_labeled']
        else:
            pct = 0
        print(f"  {action:12s}: {count:7d} ({pct:5.1f}%)")
    print()
    print("Response time matching:")
    rt = stats['response_time_stats']
    total_replies = rt['matched'] + rt['unmatched']
    if total_replies > 0:
        match_pct = 100 * rt['matched'] / total_replies
        print(f"  Matched:     {rt['matched']:7d} ({match_pct:.1f}%)")
        print(f"  Unmatched:   {rt['unmatched']:7d}")
        if rt['matched'] > 0:
            now_pct = 100 * rt['reply_now'] / rt['matched']
            later_pct = 100 * rt['reply_later'] / rt['matched']
            print(f"  REPLY_NOW:   {rt['reply_now']:7d} ({now_pct:.1f}% of matched)")
            print(f"  REPLY_LATER: {rt['reply_later']:7d} ({later_pct:.1f}% of matched)")
    print()
    print("Folder type distribution (all input):")
    for folder_type, count in sorted(stats['folder_type_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_input']
        print(f"  {folder_type:12s}: {count:7d} ({pct:5.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
