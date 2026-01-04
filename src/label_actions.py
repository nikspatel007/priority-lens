#!/usr/bin/env python3
"""Label user actions from Enron email dataset.

Infers labels from folder structure and email metadata:
- REPLIED: Subject starts with "Re:" in sent folders
- FORWARDED: Subject starts with "Fw:", "Fwd:", "FW:" in sent folders
- DELETED: Email in deleted_items folder
- ARCHIVED: Email moved to a project/personal folder
- KEPT: Email still in inbox (no explicit action)
- COMPOSED: New email in sent folders (not reply/forward)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional
from collections import Counter


# Folder classifications
SENT_FOLDERS = {'sent', 'sent_items', '_sent_mail', '_sent'}
DELETED_FOLDERS = {'deleted_items', 'connect_deletes'}
INBOX_FOLDERS = {'inbox', 'notes_inbox'}
JUNK_FOLDERS = {'junk', 'junk_file'}


def classify_folder(folder: str) -> str:
    """Classify a folder into a category.

    Returns:
        One of: 'sent', 'deleted', 'inbox', 'junk', 'archived'
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
    else:
        return 'archived'


def infer_action(email: dict) -> str:
    """Infer the user action for an email.

    Args:
        email: Email dictionary with folder, subject, etc.

    Returns:
        Action label: REPLIED, FORWARDED, DELETED, ARCHIVED, KEPT, COMPOSED, JUNK
    """
    folder = email.get('folder', '')
    subject = email.get('subject', '')
    folder_type = classify_folder(folder)

    if folder_type == 'sent':
        # This is a sent email - classify by subject
        subject_lower = subject.lower().strip()

        if subject_lower.startswith('re:'):
            return 'REPLIED'
        elif subject_lower.startswith(('fw:', 'fwd:', 'forwarded:')):
            return 'FORWARDED'
        else:
            return 'COMPOSED'

    elif folder_type == 'deleted':
        return 'DELETED'

    elif folder_type == 'inbox':
        return 'KEPT'

    elif folder_type == 'junk':
        return 'JUNK'

    else:
        # In a specific folder (project, personal, etc.)
        return 'ARCHIVED'


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

    # Count actions
    action_counts = Counter()
    folder_type_counts = Counter()

    # Add labels
    for email in emails:
        action = infer_action(email)
        email['action'] = action
        action_counts[action] += 1

        folder_type = classify_folder(email.get('folder', ''))
        folder_type_counts[folder_type] += 1

    # Write output
    print(f"Writing labeled emails to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    stats = {
        'total': len(emails),
        'action_counts': dict(action_counts),
        'folder_type_counts': dict(folder_type_counts)
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Label user actions in Enron email dataset'
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

        action_counts = Counter()
        folder_type_counts = Counter()

        for email in emails:
            action = infer_action(email)
            action_counts[action] += 1
            folder_type = classify_folder(email.get('folder', ''))
            folder_type_counts[folder_type] += 1

        stats = {
            'total': len(emails),
            'action_counts': dict(action_counts),
            'folder_type_counts': dict(folder_type_counts)
        }
    else:
        stats = process_emails(args.input, args.output, args.limit)

    # Print statistics
    print()
    print("=" * 50)
    print("LABELING STATISTICS")
    print("=" * 50)
    print(f"Total emails processed: {stats['total']}")
    print()
    print("Action distribution:")
    for action, count in sorted(stats['action_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total']
        print(f"  {action:12s}: {count:7d} ({pct:5.1f}%)")
    print()
    print("Folder type distribution:")
    for folder_type, count in sorted(stats['folder_type_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total']
        print(f"  {folder_type:12s}: {count:7d} ({pct:5.1f}%)")
    print("=" * 50)


if __name__ == '__main__':
    main()
