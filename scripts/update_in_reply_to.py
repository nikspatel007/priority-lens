#!/usr/bin/env python3
"""Update in_reply_to field in existing database records.

Reads the re-parsed JSONL and updates raw_emails and emails tables
with in_reply_to values.
"""

import asyncio
import json
import os
from pathlib import Path

import asyncpg

DB_URL = os.environ.get("DB_URL", "postgresql://postgres:postgres@localhost:5433/rl_emails")
JSONL_PATH = Path(os.environ.get("PARSED_JSONL", "/Users/nikpatel/Documents/GitHub/rl-emails/data/nik_gmail/parsed_emails.jsonl"))


def sanitize_text(text: str) -> str:
    """Remove null bytes for PostgreSQL."""
    if not text:
        return text
    return text.replace('\x00', '')


async def update_in_reply_to():
    """Update in_reply_to in existing records."""
    print(f"Reading {JSONL_PATH}...")

    # Build lookup: message_id -> (in_reply_to, references)
    lookup = {}
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                msg_id = d.get('message_id')
                in_reply_to = sanitize_text(d.get('in_reply_to', '')) or None
                references = sanitize_text(d.get('references', '')) or None
                if msg_id:
                    lookup[msg_id] = (in_reply_to, references)

    print(f"Loaded {len(lookup)} emails from JSONL")
    has_in_reply_to = sum(1 for v in lookup.values() if v[0])
    print(f"  - {has_in_reply_to} have in_reply_to values")

    # Connect and update
    print(f"\nConnecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Get message_ids that need updating
        rows = await conn.fetch("""
            SELECT id, message_id FROM raw_emails
            WHERE in_reply_to IS NULL
        """)
        print(f"\nFound {len(rows)} raw_emails records to potentially update")

        # Update in batches
        updated_raw = 0
        updated_emails = 0
        batch_size = 100

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]

            for row in batch:
                msg_id = row['message_id']
                if msg_id in lookup:
                    in_reply_to, references = lookup[msg_id]
                    if in_reply_to or references:
                        # Update raw_emails
                        await conn.execute("""
                            UPDATE raw_emails
                            SET in_reply_to = $1, references_raw = $2
                            WHERE id = $3
                        """, in_reply_to, references, row['id'])
                        updated_raw += 1

                        # Update emails
                        await conn.execute("""
                            UPDATE emails
                            SET in_reply_to = $1
                            WHERE message_id = $2
                        """, in_reply_to, msg_id)
                        updated_emails += 1

            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(rows):
                print(f"  Processed {min(i + batch_size, len(rows))}/{len(rows)}...")

        print(f"\nUpdate complete:")
        print(f"  - Updated {updated_raw} raw_emails records")
        print(f"  - Updated {updated_emails} emails records")

        # Verify
        in_reply_raw = await conn.fetchval("""
            SELECT COUNT(*) FROM raw_emails WHERE in_reply_to IS NOT NULL
        """)
        in_reply_emails = await conn.fetchval("""
            SELECT COUNT(*) FROM emails WHERE in_reply_to IS NOT NULL
        """)
        print(f"\nVerification:")
        print(f"  - raw_emails with in_reply_to: {in_reply_raw}")
        print(f"  - emails with in_reply_to: {in_reply_emails}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(update_in_reply_to())
