#!/usr/bin/env python3
"""Generate sample Gmail-like data for testing preference extraction.

Creates synthetic email data with realistic behavioral signals:
- Varied response times
- Mixed labels (starred, important, categories)
- Different actions (reply, forward, archive, delete)
- Thread structures

Usage:
    python scripts/generate_sample_gmail_data.py -o data/sample_gmail.json -n 5000

Then import to SurrealDB:
    python -m db.import_data gmail data/sample_gmail.json --url ws://localhost:8001/rpc
"""

import argparse
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path


# Sample data pools
FIRST_NAMES = [
    'Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry',
    'Ivy', 'Jack', 'Kate', 'Leo', 'Mia', 'Noah', 'Olivia', 'Peter',
]

DOMAINS = ['gmail.com', 'company.com', 'work.org', 'mail.net', 'example.com']

SUBJECTS = [
    'Meeting tomorrow', 'Quick question', 'Project update', 'FYI: {topic}',
    'Re: {topic}', 'Fw: {topic}', 'Urgent: {topic}', 'Follow up on {topic}',
    'Action required: {topic}', 'Weekly report', 'Monthly summary',
    'Invitation: {topic}', 'Reminder: {topic}', 'Thank you for {topic}',
]

TOPICS = [
    'Q1 planning', 'budget review', 'team standup', 'product launch',
    'customer feedback', 'API changes', 'deployment schedule', 'hiring',
    'performance review', 'training session', 'conference call', 'demo',
]

LABELS = [
    'INBOX', 'STARRED', 'IMPORTANT', 'UNREAD', 'CATEGORY_PERSONAL',
    'CATEGORY_UPDATES', 'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS',
    'CATEGORY_FORUMS', 'SENT', 'ARCHIVED', 'TRASH', 'SPAM',
]

ACTIONS = ['REPLY_NOW', 'REPLY_LATER', 'FORWARD', 'ARCHIVE', 'DELETE']


def generate_email_address():
    """Generate a random email address."""
    name = random.choice(FIRST_NAMES).lower()
    suffix = ''.join(random.choices(string.digits, k=2))
    domain = random.choice(DOMAINS)
    return f"{name}{suffix}@{domain}"


def generate_message_id():
    """Generate a random message ID."""
    chars = string.ascii_letters + string.digits
    uid = ''.join(random.choices(chars, k=20))
    domain = random.choice(DOMAINS)
    return f"<{uid}@{domain}>"


def generate_thread_id():
    """Generate a random Gmail thread ID."""
    return ''.join(random.choices(string.digits, k=18))


def generate_subject(is_reply=False, is_forward=False):
    """Generate a subject line."""
    topic = random.choice(TOPICS)
    template = random.choice(SUBJECTS)
    subject = template.format(topic=topic)

    if is_reply and not subject.startswith('Re:'):
        subject = f"Re: {subject}"
    elif is_forward and not subject.startswith(('Fw:', 'Fwd:')):
        subject = f"Fw: {subject}"

    return subject


def generate_body():
    """Generate email body text."""
    templates = [
        "Hi,\n\nJust wanted to follow up on this. Let me know your thoughts.\n\nBest",
        "Thanks for the update. I'll review and get back to you.\n\nCheers",
        "Can we schedule a call to discuss? I have some questions.\n\nRegards",
        "Noted. I'll add this to the agenda for our next meeting.\n\nThanks",
        "Great progress! Keep up the good work.\n\nBest regards",
        "I've attached the document you requested. Let me know if you need anything else.",
        "Please see below for the details. Happy to discuss further.",
        "Quick update: we're on track for the deadline. Will send final version soon.",
    ]
    return random.choice(templates)


def generate_date(base_date: datetime, offset_hours: int = 0):
    """Generate a formatted date string."""
    dt = base_date + timedelta(hours=offset_hours)
    # Format like: "Mon, 15 Jan 2026 14:30:00 -0800"
    return dt.strftime("%a, %d %b %Y %H:%M:%S -0800")


def generate_labels(action: str):
    """Generate labels based on action."""
    labels = ['INBOX']

    # Actions influence label probability
    if action == 'DELETE':
        if random.random() < 0.3:
            labels = ['TRASH']
        elif random.random() < 0.2:
            labels = ['SPAM']
    elif action == 'ARCHIVE':
        labels = ['ARCHIVED']
    elif action in ('REPLY_NOW', 'REPLY_LATER'):
        labels.append('SENT')

    # Add importance labels
    if action == 'REPLY_NOW' and random.random() < 0.6:
        labels.append('IMPORTANT')
        if random.random() < 0.5:
            labels.append('STARRED')
    elif action == 'REPLY_LATER' and random.random() < 0.3:
        labels.append('IMPORTANT')
    elif action == 'FORWARD' and random.random() < 0.2:
        labels.append('IMPORTANT')

    # Add category labels
    if random.random() < 0.3:
        cat = random.choice([
            'CATEGORY_PERSONAL', 'CATEGORY_UPDATES',
            'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS',
        ])
        labels.append(cat)

    return list(set(labels))


def generate_emails(
    count: int,
    start_date: datetime,
    seed: int = 42
) -> list[dict]:
    """Generate synthetic email dataset.

    Args:
        count: Number of emails to generate
        start_date: Starting date for email timestamps
        seed: Random seed for reproducibility

    Returns:
        List of email dictionaries
    """
    random.seed(seed)

    emails = []
    threads = {}  # thread_id -> list of (message_id, date, email_dict)
    message_index = {}  # message_id -> email

    # Pre-generate some email addresses for realistic patterns
    user_email = generate_email_address()
    contacts = [generate_email_address() for _ in range(50)]

    current_date = start_date

    for i in range(count):
        # Decide if this is part of an existing thread
        is_thread_reply = len(threads) > 0 and random.random() < 0.4
        is_forward = random.random() < 0.1

        if is_thread_reply:
            thread_id = random.choice(list(threads.keys()))
            # Pick a random message from this thread to reply to
            thread_msgs = threads[thread_id]
            parent_msg_id, parent_date, parent_email = random.choice(thread_msgs)
            # Strip angle brackets for matching
            in_reply_to = parent_msg_id.strip('<>')
            subject = generate_subject(is_reply=True)
        else:
            thread_id = generate_thread_id()
            in_reply_to = None
            subject = generate_subject(is_forward=is_forward)

        # Generate action with realistic distribution
        # If it's a reply, bias toward reply actions
        if is_thread_reply:
            action_weights = {
                'DELETE': 0.10,
                'ARCHIVE': 0.15,
                'REPLY_LATER': 0.35,
                'REPLY_NOW': 0.30,
                'FORWARD': 0.10,
            }
        else:
            action_weights = {
                'DELETE': 0.50,
                'ARCHIVE': 0.30,
                'REPLY_LATER': 0.08,
                'REPLY_NOW': 0.05,
                'FORWARD': 0.07,
            }
        action = random.choices(
            list(action_weights.keys()),
            weights=list(action_weights.values())
        )[0]

        # Generate message
        message_id = generate_message_id()
        from_email = random.choice(contacts) if random.random() < 0.7 else user_email
        to_emails = [user_email] if from_email != user_email else [random.choice(contacts)]

        # Advance time - replies are faster
        if is_thread_reply and action == 'REPLY_NOW':
            hours_delta = random.uniform(0.1, 1.0)  # < 1 hour
        elif is_thread_reply and action == 'REPLY_LATER':
            hours_delta = random.uniform(2, 48)  # 2-48 hours
        else:
            hours_delta = random.randint(1, 24)
        current_date += timedelta(hours=hours_delta)

        email = {
            'message_id': message_id,
            'date': generate_date(current_date),
            'from': from_email,
            'to': ', '.join(to_emails),
            'cc': '',
            'bcc': '',
            'subject': subject,
            'body': generate_body(),
            'in_reply_to': in_reply_to,
            'references': in_reply_to or '',
            'thread_id': thread_id,
            'labels': generate_labels(action),
            'action': action.lower(),
        }

        emails.append(email)

        # Track thread membership with proper message_id for reply matching
        if thread_id not in threads:
            threads[thread_id] = []
        threads[thread_id].append((message_id, current_date, email))
        message_index[message_id] = email

        # Limit thread depth
        if len(threads[thread_id]) > 10:
            del threads[thread_id]

        # Clean old threads
        if len(threads) > 100:
            oldest_thread = list(threads.keys())[0]
            del threads[oldest_thread]

    return emails


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample Gmail-like data for testing'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('data/sample_gmail.json'),
        help='Output JSON file (default: data/sample_gmail.json)'
    )
    parser.add_argument(
        '-n', '--count',
        type=int,
        default=5000,
        help='Number of emails to generate (default: 5000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    start_date = datetime(2025, 1, 1, 9, 0, 0)

    print(f"Generating {args.count} synthetic emails...")
    emails = generate_emails(args.count, start_date, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(emails, f, indent=2)

    print(f"Wrote {len(emails)} emails to {args.output}")

    # Print statistics
    from collections import Counter
    actions = Counter(e['action'] for e in emails)
    print("\nAction distribution:")
    for action, count in actions.most_common():
        pct = 100 * count / len(emails)
        print(f"  {action}: {count} ({pct:.1f}%)")

    label_counts = Counter()
    for e in emails:
        for label in e['labels']:
            label_counts[label] += 1
    print("\nLabel distribution (top 10):")
    for label, count in label_counts.most_common(10):
        pct = 100 * count / len(emails)
        print(f"  {label}: {count} ({pct:.1f}%)")

    threads = Counter(e['thread_id'] for e in emails)
    print(f"\nTotal threads: {len(threads)}")
    print(f"Avg emails per thread: {len(emails) / len(threads):.1f}")


if __name__ == '__main__':
    main()
