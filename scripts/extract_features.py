#!/usr/bin/env python3
"""
Extract ML features from all emails and store in email_features table.

Runs the combined feature extraction pipeline on all emails in the database,
storing the resulting feature vectors and scores for ML training.

Usage:
    python scripts/extract_features.py [--batch-size 500] [--skip-content]
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    CombinedFeatureExtractor,
    extract_combined_features,
)

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"
DEFAULT_BATCH_SIZE = 500
EXTRACTION_VERSION = 1


async def get_email_count(pool: asyncpg.Pool) -> int:
    """Get total number of emails to process."""
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT COUNT(*) FROM emails")
        return result


async def get_processed_count(pool: asyncpg.Pool) -> int:
    """Get number of emails already processed."""
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT COUNT(*) FROM email_features")
        return result


async def fetch_emails_batch(
    pool: asyncpg.Pool,
    offset: int,
    limit: int,
) -> list[dict]:
    """Fetch a batch of emails from the database."""
    query = """
        SELECT
            e.id,
            e.message_id,
            e.from_email,
            e.from_name,
            e.to_emails,
            e.cc_emails,
            e.subject,
            e.body_text,
            e.date_parsed,
            e.labels,
            e.is_sent
        FROM emails e
        LEFT JOIN email_features ef ON e.id = ef.email_id
        WHERE ef.id IS NULL
        ORDER BY e.id
        LIMIT $1 OFFSET $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit, offset)
        return [dict(row) for row in rows]


def convert_to_email_dict(row: dict) -> dict:
    """Convert database row to email dict format expected by feature extractor."""
    # Convert array fields to comma-separated strings (extractor expects strings)
    to_emails = row['to_emails'] or []
    cc_emails = row['cc_emails'] or []

    return {
        'id': row['id'],
        'message_id': row['message_id'],
        'from': row['from_email'] or '',
        'from_name': row['from_name'] or '',
        'to': ', '.join(to_emails) if isinstance(to_emails, list) else str(to_emails or ''),
        'cc': ', '.join(cc_emails) if isinstance(cc_emails, list) else str(cc_emails or ''),
        'subject': row['subject'] or '',
        'body': row['body_text'] or '',
        'date': row['date_parsed'].isoformat() if row['date_parsed'] else '',
        'labels': row['labels'] or [],
        'is_sent': row['is_sent'] or False,
    }


async def insert_features_batch(
    pool: asyncpg.Pool,
    features_batch: list[tuple],
) -> int:
    """Insert a batch of feature records."""
    query = """
        INSERT INTO email_features (
            email_id,
            project_score,
            topic_score,
            task_score,
            people_score,
            temporal_score,
            service_score,
            relationship_score,
            overall_priority,
            feature_vector,
            feature_dim,
            content_embedding,
            content_dim,
            extraction_version
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (email_id) DO NOTHING
    """
    async with pool.acquire() as conn:
        await conn.executemany(query, features_batch)
        return len(features_batch)


def extract_features_for_batch(
    emails: list[dict],
    include_content: bool = False,
) -> list[tuple]:
    """Extract features for a batch of emails."""
    results = []

    for email in emails:
        email_dict = convert_to_email_dict(email)

        try:
            features = extract_combined_features(
                email_dict,
                user_email='',  # Will be inferred from is_sent
                include_content=include_content,
            )

            # Get feature vector
            feature_vec = features.to_feature_vector(include_content=False)
            feature_list = list(feature_vec) if hasattr(feature_vec, '__iter__') else [float(feature_vec)]

            # Get content embedding if available
            content_emb = None
            content_dim = None
            if include_content and features.content is not None:
                content_vec = features.content.to_feature_vector()
                content_emb = list(content_vec) if hasattr(content_vec, '__iter__') else None
                content_dim = len(content_emb) if content_emb else None

            results.append((
                email['id'],
                float(features.project_score),
                float(features.topic_score),
                float(features.task_score),
                float(features.people_score),
                float(features.temporal_score),
                float(features.service_score),
                float(features.relationship_score),
                float(features.overall_priority),
                feature_list,
                len(feature_list),
                content_emb,
                content_dim,
                EXTRACTION_VERSION,
            ))
        except Exception as e:
            print(f"  Warning: Failed to extract features for email {email['id']}: {e}")
            continue

    return results


async def run_extraction(
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_content: bool = False,
    max_batches: Optional[int] = None,
) -> dict:
    """Run feature extraction on all emails."""
    print(f"Connecting to database...")
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)

    try:
        total_emails = await get_email_count(pool)
        processed = await get_processed_count(pool)
        remaining = total_emails - processed

        print(f"Total emails: {total_emails}")
        print(f"Already processed: {processed}")
        print(f"Remaining: {remaining}")
        print(f"Batch size: {batch_size}")
        print(f"Include content embeddings: {include_content}")
        print()

        if remaining == 0:
            print("All emails already processed!")
            return {'processed': 0, 'total': total_emails, 'errors': 0}

        start_time = time.time()
        total_processed = 0
        total_errors = 0
        batch_num = 0

        while True:
            batch_num += 1
            if max_batches and batch_num > max_batches:
                print(f"\nReached max batches limit ({max_batches})")
                break

            # Fetch batch (always offset 0 since we skip already processed)
            emails = await fetch_emails_batch(pool, 0, batch_size)

            if not emails:
                break

            # Extract features
            batch_start = time.time()
            features_batch = extract_features_for_batch(emails, include_content)

            # Insert to database
            if features_batch:
                inserted = await insert_features_batch(pool, features_batch)
                total_processed += inserted
                total_errors += len(emails) - len(features_batch)

            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            eta = (remaining - total_processed) / rate if rate > 0 else 0

            print(
                f"Batch {batch_num}: {len(features_batch)}/{len(emails)} extracted | "
                f"Total: {total_processed}/{remaining} | "
                f"Rate: {rate:.1f}/s | "
                f"ETA: {eta/60:.1f}m | "
                f"Batch time: {batch_time:.1f}s"
            )

        elapsed = time.time() - start_time
        print()
        print(f"Extraction complete!")
        print(f"  Processed: {total_processed}")
        print(f"  Errors: {total_errors}")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Rate: {total_processed/elapsed:.1f} emails/second")

        return {
            'processed': total_processed,
            'total': total_emails,
            'errors': total_errors,
            'time_seconds': elapsed,
        }

    finally:
        await pool.close()


async def verify_extraction(pool: asyncpg.Pool) -> dict:
    """Verify extraction results."""
    async with pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total,
                AVG(overall_priority) as avg_priority,
                MIN(overall_priority) as min_priority,
                MAX(overall_priority) as max_priority,
                AVG(feature_dim) as avg_dim
            FROM email_features
        """)
        return dict(stats)


def main():
    parser = argparse.ArgumentParser(description='Extract ML features from emails')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for processing (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--include-content', action='store_true',
                        help='Include content embeddings (requires sentence-transformers)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to process (for testing)')
    args = parser.parse_args()

    print("=" * 60)
    print("EMAIL FEATURE EXTRACTION")
    print("=" * 60)
    print()

    result = asyncio.run(run_extraction(
        batch_size=args.batch_size,
        include_content=args.include_content,
        max_batches=args.max_batches,
    ))

    print()
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Processed: {result['processed']}")
    print(f"  Errors: {result['errors']}")
    if 'time_seconds' in result:
        print(f"  Time: {result['time_seconds']/60:.1f} minutes")


if __name__ == '__main__':
    main()
