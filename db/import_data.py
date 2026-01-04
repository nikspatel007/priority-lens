#!/usr/bin/env python3
"""Import email data into SurrealDB.

This script imports Enron or Gmail email data from JSON files into SurrealDB,
creating users, threads, and communication graph edges.

Usage:
    # Start SurrealDB first:
    surreal start file:data/enron.db --user root --pass root --bind 0.0.0.0:8000

    # Import Enron data:
    python -m db.import_data enron data/train.json data/val.json data/test.json

    # Import Gmail data:
    python -m db.import_data gmail data/gmail_emails.json --labels
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from surrealdb import AsyncSurreal


# Email address extraction regex
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')


def extract_emails(field: str) -> list[str]:
    """Extract email addresses from a header field."""
    if not field:
        return []
    return EMAIL_PATTERN.findall(field.lower())


def parse_date(date_str: str) -> Optional[str]:
    """Parse date string to ISO format for SurrealDB."""
    if not date_str:
        return None

    # Common date formats in Enron/Gmail
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%d %b %Y %H:%M:%S %z',
        '%Y-%m-%d %H:%M:%S',
        '%a %b %d %H:%M:%S %Y',
    ]

    # Clean up the date string
    date_str = date_str.strip()
    # Remove extra timezone info like "(PST)"
    date_str = re.sub(r'\s*\([A-Z]+\)\s*$', '', date_str)

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Validate year is reasonable (Enron emails are 1999-2002)
            if dt.year < 1990 or dt.year > 2030:
                continue  # Invalid year, try next format
            return dt.isoformat()
        except ValueError:
            continue

    return None


def extract_org_level(x_from: str) -> Optional[str]:
    """Extract org level from Enron X-From header.

    Example: "Smith, John (Vice President, Trading)" -> "Vice President, Trading"
    """
    if not x_from:
        return None

    match = re.search(r'\(([^)]+)\)', x_from)
    if match:
        return match.group(1)
    return None


class EmailImporter:
    """Import email data into SurrealDB."""

    def __init__(self, url: str, namespace: str, database: str):
        self.url = url
        self.namespace = namespace
        self.database = database
        self.db: AsyncSurreal = None
        self.users_cache: dict[str, str] = {}  # email -> record id
        self.threads_cache: dict[str, str] = {}  # thread_id -> record id
        self.emails_cache: dict[str, str] = {}  # message_id -> record id
        self.comm_edges: dict[tuple[str, str], dict] = defaultdict(
            lambda: {'email_count': 0, 'reply_count': 0}
        )

    async def connect(self, username: str, password: str):
        """Connect to SurrealDB."""
        self.db = AsyncSurreal(self.url)
        await self.db.signin({'username': username, 'password': password})
        await self.db.use(self.namespace, self.database)

    async def import_schema(self, schema_path: Path):
        """Import schema from .surql file."""
        schema = schema_path.read_text()
        # Execute schema statements one by one
        for statement in schema.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    await self.db.query(statement)
                except Exception as e:
                    print(f"Warning: Schema statement failed: {e}")

    async def get_or_create_user(
        self,
        email: str,
        name: Optional[str] = None,
        org_level: Optional[str] = None,
    ) -> str:
        """Get or create a user record, return record ID."""
        email = email.lower().strip()

        if email in self.users_cache:
            return self.users_cache[email]

        # Check if user exists
        result = await self.db.query(
            'SELECT id FROM users WHERE email = $email LIMIT 1',
            {'email': email}
        )

        # New API returns list directly, not nested under 'result'
        if result and isinstance(result, list) and len(result) > 0:
            user_id = str(result[0]['id'])
            self.users_cache[email] = user_id
            return user_id

        # Create new user
        user_data = {
            'email': email,
            'name': name,
            'org_level': org_level,
            'is_internal': '@enron.com' in email,
        }

        result = await self.db.create('users', user_data)
        # Check if result is error string
        if isinstance(result, str):
            raise ValueError(f"Failed to create user: {result}")
        user_id = str(result['id'])
        self.users_cache[email] = user_id
        return user_id

    async def get_or_create_thread(
        self,
        thread_id: str,
        subject: str = '',
    ) -> str:
        """Get or create a thread record, return record ID."""
        if thread_id in self.threads_cache:
            return self.threads_cache[thread_id]

        # Check if thread exists
        result = await self.db.query(
            'SELECT id FROM threads WHERE thread_id = $thread_id LIMIT 1',
            {'thread_id': thread_id}
        )

        # New API returns list directly
        if result and isinstance(result, list) and len(result) > 0:
            rec_id = str(result[0]['id'])
            self.threads_cache[thread_id] = rec_id
            return rec_id

        # Create new thread
        thread_data = {
            'thread_id': thread_id,
            'subject': subject,
        }

        result = await self.db.create('threads', thread_data)
        if isinstance(result, str):
            raise ValueError(f"Failed to create thread: {result}")
        rec_id = str(result['id'])
        self.threads_cache[thread_id] = rec_id
        return rec_id

    async def import_email(
        self,
        email_data: dict,
        source: str = 'enron',
        split: Optional[str] = None,
    ) -> Optional[str]:
        """Import a single email record."""
        message_id = email_data.get('message_id', '')
        if not message_id:
            return None

        # Check if already exists
        if message_id in self.emails_cache:
            return self.emails_cache[message_id]

        # Parse fields
        from_email = extract_emails(email_data.get('from', ''))
        from_email = from_email[0] if from_email else ''

        to_emails = extract_emails(email_data.get('to', ''))
        cc_emails = extract_emails(email_data.get('cc', ''))
        bcc_emails = extract_emails(email_data.get('bcc', ''))

        date_str = email_data.get('date', '')

        # Build email record (skip date field as SurrealDB datetime parsing is complex)
        record = {
            'message_id': message_id,
            'date_str': date_str,
            'subject': email_data.get('subject', ''),
            'body': email_data.get('body', ''),
            'from_email': from_email,
            'to_emails': to_emails,
            'cc_emails': cc_emails,
            'bcc_emails': bcc_emails,
            'in_reply_to': email_data.get('in_reply_to'),
            'split': split,
            'action': email_data.get('action'),
            'timing': email_data.get('timing'),
        }

        # Add source-specific fields
        if source == 'enron':
            record['x_from'] = email_data.get('x_from')
            record['x_to'] = email_data.get('x_to')
            record['x_folder'] = email_data.get('x_folder')
            record['x_origin'] = email_data.get('x_origin')
            record['folder'] = email_data.get('folder')
            record['file_path'] = email_data.get('file_path')
            record['attachments'] = email_data.get('attachments', [])
            record['enron_user'] = email_data.get('user')
        elif source == 'gmail':
            record['gmail_thread_id'] = email_data.get('thread_id')
            record['labels'] = email_data.get('labels', [])

        # Parse references
        refs = email_data.get('references', '')
        if refs:
            record['references'] = EMAIL_PATTERN.findall(refs)

        try:
            result = await self.db.create('emails', record)
            # Check if result is error string
            if isinstance(result, str):
                raise ValueError(f"Failed to create email: {result}")
            email_id = str(result['id'])
            self.emails_cache[message_id] = email_id

            # Create user records and relationships
            if from_email:
                org_level = extract_org_level(email_data.get('x_from', ''))
                sender_id = await self.get_or_create_user(from_email, org_level=org_level)
                await self.db.query(
                    f'RELATE {email_id}->sent_by->{sender_id}'
                )

                # Track communication edges
                for to_email in to_emails:
                    self.comm_edges[(from_email, to_email)]['email_count'] += 1

            # Create recipient relationships
            for to_email in to_emails:
                recipient_id = await self.get_or_create_user(to_email)
                await self.db.query(
                    f'RELATE {email_id}->received_by->{recipient_id} SET field = "to"'
                )

            for cc_email in cc_emails:
                recipient_id = await self.get_or_create_user(cc_email)
                await self.db.query(
                    f'RELATE {email_id}->received_by->{recipient_id} SET field = "cc"'
                )

            # Handle threading
            thread_id = email_data.get('thread_id') or email_data.get('in_reply_to') or message_id
            if thread_id:
                thread_rec_id = await self.get_or_create_thread(
                    thread_id, email_data.get('subject', '')
                )
                await self.db.query(
                    f'RELATE {email_id}->belongs_to->{thread_rec_id}'
                )

            # Handle reply relationships
            in_reply_to = email_data.get('in_reply_to')
            if in_reply_to and in_reply_to in self.emails_cache:
                parent_id = self.emails_cache[in_reply_to]
                await self.db.query(
                    f'RELATE {email_id}->replies_to->{parent_id}'
                )
                # Track reply in comm edges
                if from_email:
                    for to_email in to_emails:
                        self.comm_edges[(from_email, to_email)]['reply_count'] += 1

            return email_id

        except Exception as e:
            print(f"Error importing email {message_id}: {e}", file=sys.stderr)
            return None

    async def finalize_comm_edges(self):
        """Create communication graph edges from accumulated data."""
        print(f"Creating {len(self.comm_edges)} communication edges...")

        for (from_email, to_email), stats in self.comm_edges.items():
            if from_email not in self.users_cache or to_email not in self.users_cache:
                continue

            from_id = self.users_cache[from_email]
            to_id = self.users_cache[to_email]

            try:
                await self.db.query(
                    f'''RELATE {from_id}->communicates->{to_id} SET
                        email_count = $email_count,
                        reply_count = $reply_count,
                        updated_at = time::now()
                    ''',
                    stats
                )
            except Exception as e:
                print(f"Warning: Could not create comm edge: {e}")

    async def import_json_file(
        self,
        json_path: Path,
        source: str = 'enron',
        split: Optional[str] = None,
        batch_size: int = 100,
    ) -> int:
        """Import emails from a JSON file."""
        print(f"Loading {json_path}...")

        with open(json_path, 'r', encoding='utf-8') as f:
            emails = json.load(f)

        print(f"Importing {len(emails)} emails (split={split})...")

        imported = 0
        for i, email_data in enumerate(emails):
            result = await self.import_email(email_data, source=source, split=split)
            if result:
                imported += 1

            if (i + 1) % batch_size == 0:
                print(f"  Processed {i + 1}/{len(emails)} emails...")

        print(f"  Imported {imported}/{len(emails)} emails")
        return imported


async def main():
    parser = argparse.ArgumentParser(
        description='Import email data into SurrealDB'
    )
    parser.add_argument(
        'source',
        choices=['enron', 'gmail'],
        help='Data source type'
    )
    parser.add_argument(
        'files',
        type=Path,
        nargs='+',
        help='JSON files to import (for Enron: train.json val.json test.json)'
    )
    parser.add_argument(
        '--url',
        default='ws://localhost:8000/rpc',
        help='SurrealDB connection URL'
    )
    parser.add_argument(
        '--user',
        default='root',
        help='SurrealDB username'
    )
    parser.add_argument(
        '--pass',
        dest='password',
        default='root',
        help='SurrealDB password'
    )
    parser.add_argument(
        '--namespace',
        default='rl_emails',
        help='SurrealDB namespace'
    )
    parser.add_argument(
        '--database',
        help='SurrealDB database name (default: same as source)'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default=Path(__file__).parent / 'schema.surql',
        help='Path to schema file'
    )
    parser.add_argument(
        '--skip-schema',
        action='store_true',
        help='Skip schema import (if already imported)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for progress reporting'
    )

    args = parser.parse_args()

    database = args.database or args.source

    # Connect to SurrealDB
    importer = EmailImporter(args.url, args.namespace, database)

    print(f"Connecting to {args.url}...")
    await importer.connect(args.user, args.password)
    print(f"Connected to namespace={args.namespace}, database={database}")

    # Import schema if not skipping
    if not args.skip_schema and args.schema.exists():
        print(f"Importing schema from {args.schema}...")
        await importer.import_schema(args.schema)
        print("Schema imported")

    # Import files
    total_imported = 0

    if args.source == 'enron':
        # For Enron, expect train.json, val.json, test.json
        split_map = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }

        for json_file in args.files:
            # Determine split from filename
            split = None
            for key in split_map:
                if key in json_file.name:
                    split = split_map[key]
                    break

            count = await importer.import_json_file(
                json_file,
                source='enron',
                split=split,
                batch_size=args.batch_size,
            )
            total_imported += count

    else:  # gmail
        for json_file in args.files:
            count = await importer.import_json_file(
                json_file,
                source='gmail',
                split=None,  # Gmail doesn't have predefined splits
                batch_size=args.batch_size,
            )
            total_imported += count

    # Finalize communication edges
    await importer.finalize_comm_edges()

    print(f"\nImport complete!")
    print(f"  Total emails: {total_imported}")
    print(f"  Total users: {len(importer.users_cache)}")
    print(f"  Total threads: {len(importer.threads_cache)}")
    print(f"  Communication edges: {len(importer.comm_edges)}")

    await importer.db.close()


if __name__ == '__main__':
    asyncio.run(main())
