#!/usr/bin/env python3
"""
Stage 4: Clean and Normalize Email Data

Cleaning Operations:
- Normalize all dates to ISO 8601 UTC
- Deduplicate by message_id (keep latest)
- Fix common encoding issues (UTF-8 normalization)
- Standardize email addresses (lowercase, trim)
- Normalize label names (uppercase, trim)

Filtering:
- Remove emails with unparseable dates
- Remove emails missing critical fields (message_id, from, date)

Output:
- cleaned_emails.jsonl
- cleaning_log.json
"""

import json
import re
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional


INPUT_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/parsed_emails.jsonl")
OUTPUT_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/cleaned_emails.jsonl")
LOG_PATH = Path("/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/cleaning_log.json")


# Email regex pattern for validation
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)


def normalize_date(date_str: str) -> Optional[str]:
    """Convert RFC 2822 date to ISO 8601 UTC format."""
    if not date_str or not date_str.strip():
        return None

    try:
        # Parse RFC 2822 date
        dt = parsedate_to_datetime(date_str)
        # Convert to UTC
        dt_utc = dt.astimezone(timezone.utc)
        # Return ISO 8601 format
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        # Try some common alternative formats
        alt_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S %z',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%d %b %Y %H:%M:%S',
        ]
        for fmt in alt_formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_utc = dt.astimezone(timezone.utc)
                return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
            except Exception:
                continue
        return None


def extract_email_address(addr_str: str) -> Optional[str]:
    """Extract email address from 'Name <email>' format."""
    if not addr_str:
        return None

    addr_str = addr_str.strip()

    # Try to extract from angle brackets
    match = re.search(r'<([^>]+)>', addr_str)
    if match:
        email = match.group(1).strip().lower()
    else:
        # Assume it's just an email address
        email = addr_str.lower()

    return email if email else None


def normalize_email_field(field: str) -> str:
    """Normalize an email field (may contain multiple addresses)."""
    if not field:
        return ""

    # Normalize unicode
    field = unicodedata.normalize('NFC', field)

    # Extract and normalize each email address
    # Split by comma, keeping the structure
    parts = []
    for part in field.split(','):
        part = part.strip()
        if not part:
            continue

        # Extract name and email
        match = re.match(r'^(.*?)\s*<([^>]+)>\s*$', part)
        if match:
            name = match.group(1).strip()
            email = match.group(2).strip().lower()
            if name:
                parts.append(f"{name} <{email}>")
            else:
                parts.append(f"<{email}>")
        else:
            # Just an email address
            parts.append(part.lower().strip())

    return ", ".join(parts)


def normalize_labels(labels: list) -> list:
    """Normalize label names: uppercase, trim."""
    if not labels:
        return []

    normalized = []
    for label in labels:
        if isinstance(label, str):
            # Normalize unicode
            label = unicodedata.normalize('NFC', label)
            # Uppercase and trim
            label = label.strip().upper()
            if label:
                normalized.append(label)

    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for label in normalized:
        if label not in seen:
            seen.add(label)
            deduped.append(label)

    return deduped


def normalize_text(text: str) -> str:
    """Normalize text: fix encoding issues, normalize unicode."""
    if not text:
        return ""

    # Normalize unicode (NFC form for consistency)
    text = unicodedata.normalize('NFC', text)

    # Fix common mojibake patterns
    mojibake_fixes = [
        ('â€™', "'"),
        ('â€œ', '"'),
        ('â€', '"'),
        ('â€"', '—'),
        ('â€"', '–'),
        ('Ã©', 'é'),
        ('Ã¨', 'è'),
        ('Ã ', 'à'),
        ('Ã¢', 'â'),
        ('Ã®', 'î'),
        ('Ã´', 'ô'),
        ('Ã»', 'û'),
        ('Ã§', 'ç'),
        ('Ã¯', 'ï'),
        ('Ã«', 'ë'),
        ('Ã¼', 'ü'),
        ('Ã¶', 'ö'),
        ('Ã¤', 'ä'),
        ('Â°', '°'),
        ('Â©', '©'),
        ('Â®', '®'),
        ('\x00', ''),  # Remove null bytes
    ]

    for bad, good in mojibake_fixes:
        text = text.replace(bad, good)

    return text


def validate_email(email_str: str) -> bool:
    """Validate email address format."""
    if not email_str:
        return False
    return EMAIL_PATTERN.match(email_str) is not None


def clean_email(email: dict, log: dict) -> Optional[dict]:
    """Clean a single email record. Returns None if invalid."""

    # Check critical fields
    message_id = email.get('message_id', '').strip()
    from_field = email.get('from', '').strip()
    date_field = email.get('date', '').strip()

    if not message_id:
        log['missing_message_id'] += 1
        return None

    if not from_field:
        log['missing_from'] += 1
        return None

    if not date_field:
        log['missing_date'] += 1
        return None

    # Normalize date
    normalized_date = normalize_date(date_field)
    if normalized_date is None:
        log['unparseable_date'] += 1
        log['unparseable_date_samples'].append({
            'message_id': message_id,
            'date': date_field
        })
        return None

    # Normalize all fields
    cleaned = {
        'message_id': message_id,
        'thread_id': email.get('thread_id', '').strip(),
        'labels': normalize_labels(email.get('labels', [])),
        'date': normalized_date,
        'date_original': date_field,  # Keep original for reference
        'from': normalize_email_field(from_field),
        'to': normalize_email_field(email.get('to', '')),
        'cc': normalize_email_field(email.get('cc', '')),
        'subject': normalize_text(email.get('subject', '')),
        'body': normalize_text(email.get('body', '')),
        'has_attachments': email.get('has_attachments', False),
    }

    # Extract sender email for validation
    sender_email = extract_email_address(cleaned['from'])
    if not sender_email or not validate_email(sender_email):
        log['invalid_from_email'] += 1
        log['invalid_from_samples'].append({
            'message_id': message_id,
            'from': cleaned['from']
        })
        # Don't reject, just log - some old emails have weird formats

    return cleaned


def deduplicate_emails(emails: list, log: dict) -> list:
    """Deduplicate by message_id, keeping the latest (by date)."""
    by_message_id = defaultdict(list)

    for email in emails:
        by_message_id[email['message_id']].append(email)

    deduped = []
    for msg_id, dupes in by_message_id.items():
        if len(dupes) > 1:
            log['duplicates_removed'] += len(dupes) - 1
            log['duplicate_message_ids'].append({
                'message_id': msg_id,
                'count': len(dupes)
            })
            # Keep the latest by date
            dupes.sort(key=lambda x: x['date'], reverse=True)
        deduped.append(dupes[0])

    return deduped


def main():
    print(f"Stage 4: Clean and Normalize Email Data")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Initialize log
    log = {
        'input_count': 0,
        'output_count': 0,
        'missing_message_id': 0,
        'missing_from': 0,
        'missing_date': 0,
        'unparseable_date': 0,
        'unparseable_date_samples': [],
        'invalid_from_email': 0,
        'invalid_from_samples': [],
        'duplicates_removed': 0,
        'duplicate_message_ids': [],
        'encoding_fixes': 0,
    }

    # Read and clean emails
    cleaned_emails = []

    print("Reading and cleaning emails...")
    with open(INPUT_PATH, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i % 5000 == 0:
                print(f"  Processed {i} emails...")

            log['input_count'] += 1

            try:
                email = json.loads(line)
            except json.JSONDecodeError:
                log['json_parse_errors'] = log.get('json_parse_errors', 0) + 1
                continue

            cleaned = clean_email(email, log)
            if cleaned:
                cleaned_emails.append(cleaned)

    print(f"  Total read: {log['input_count']}")
    print(f"  After cleaning: {len(cleaned_emails)}")

    # Deduplicate
    print("\nDeduplicating by message_id...")
    cleaned_emails = deduplicate_emails(cleaned_emails, log)
    print(f"  After deduplication: {len(cleaned_emails)}")

    # Sort by date
    print("\nSorting by date...")
    cleaned_emails.sort(key=lambda x: x['date'])

    log['output_count'] = len(cleaned_emails)

    # Limit sample arrays in log
    log['unparseable_date_samples'] = log['unparseable_date_samples'][:20]
    log['invalid_from_samples'] = log['invalid_from_samples'][:20]
    log['duplicate_message_ids'] = log['duplicate_message_ids'][:20]

    # Write output
    print(f"\nWriting {len(cleaned_emails)} cleaned emails...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for email in cleaned_emails:
            f.write(json.dumps(email, ensure_ascii=False) + '\n')

    # Write log
    print(f"Writing cleaning log...")
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Input emails:        {log['input_count']:,}")
    print(f"Output emails:       {log['output_count']:,}")
    print(f"Removed (total):     {log['input_count'] - log['output_count']:,}")
    print()
    print("Removal reasons:")
    print(f"  Missing message_id: {log['missing_message_id']:,}")
    print(f"  Missing from:       {log['missing_from']:,}")
    print(f"  Missing date:       {log['missing_date']:,}")
    print(f"  Unparseable date:   {log['unparseable_date']:,}")
    print(f"  Duplicates:         {log['duplicates_removed']:,}")
    print()
    print(f"Warnings:")
    print(f"  Invalid from email: {log['invalid_from_email']:,}")
    print()

    # Validation
    print("="*60)
    print("VALIDATION")
    print("="*60)

    # Check for duplicate message_ids
    message_ids = [e['message_id'] for e in cleaned_emails]
    unique_ids = set(message_ids)
    if len(message_ids) == len(unique_ids):
        print("[PASS] No duplicate message_ids")
    else:
        print(f"[FAIL] {len(message_ids) - len(unique_ids)} duplicate message_ids remain")

    # Check all dates parseable (they should be since we normalized them)
    date_check_passed = all(
        e['date'] and 'T' in e['date'] and e['date'].endswith('Z')
        for e in cleaned_emails
    )
    if date_check_passed:
        print("[PASS] All dates in ISO 8601 UTC format")
    else:
        print("[FAIL] Some dates not in expected format")

    # Check email format
    from_emails_valid = 0
    for e in cleaned_emails:
        email_addr = extract_email_address(e['from'])
        if email_addr and validate_email(email_addr):
            from_emails_valid += 1

    validity_pct = (from_emails_valid / len(cleaned_emails)) * 100 if cleaned_emails else 0
    print(f"[INFO] Valid 'from' email format: {from_emails_valid:,}/{len(cleaned_emails):,} ({validity_pct:.1f}%)")

    print()
    print(f"Output files:")
    print(f"  {OUTPUT_PATH}")
    print(f"  {LOG_PATH}")


if __name__ == '__main__':
    main()
