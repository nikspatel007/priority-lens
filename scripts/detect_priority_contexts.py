#!/usr/bin/env python3
"""Detect priority contexts via response time analysis.

Analyzes email response times to find periods of heightened engagement.
Periods with significantly faster response times indicate active priority contexts.

Usage:
    uv run python scripts/detect_priority_contexts.py
    uv run python scripts/detect_priority_contexts.py --dry-run
    uv run python scripts/detect_priority_contexts.py --threshold 1.5
"""

import argparse
import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Optional

import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/rl_emails')

# Default threshold: periods with avg response time this many std devs below mean
DEFAULT_THRESHOLD = 1.0

# Minimum emails in a period to consider it
MIN_EMAILS_PER_PERIOD = 5


@dataclass
class TimePeriod:
    """A time period with response statistics."""
    start: datetime
    end: datetime
    email_count: int
    avg_response_seconds: float
    median_response_seconds: float
    key_participants: list[str]
    sample_subjects: list[str]

    @property
    def period_name(self) -> str:
        """Generate a name for this period."""
        return f"High engagement: {self.start.strftime('%b %Y')}"

    def z_score(self, global_mean: float, global_std: float) -> float:
        """Calculate how many std devs below mean this period is."""
        if global_std == 0:
            return 0
        return (global_mean - self.avg_response_seconds) / global_std


async def get_response_data(conn: asyncpg.Connection) -> list[dict]:
    """Fetch all emails with response time data.

    If response_time_seconds is not populated, compute it on the fly
    by matching sent emails to received emails via in_reply_to.
    """
    print("Fetching response time data...")

    # First check if we have pre-computed response times
    precomputed_count = await conn.fetchval("""
        SELECT COUNT(*) FROM emails WHERE response_time_seconds IS NOT NULL
    """)

    if precomputed_count > 0:
        print(f"Using {precomputed_count} pre-computed response times")
        rows = await conn.fetch("""
            SELECT
                id,
                date_parsed,
                response_time_seconds,
                from_email,
                to_emails,
                subject,
                'precomputed' as timing
            FROM emails
            WHERE response_time_seconds IS NOT NULL
              AND date_parsed IS NOT NULL
            ORDER BY date_parsed
        """)
        return [dict(r) for r in rows]

    # Compute response times within threads
    # Find sent emails that follow received emails in the same thread
    print("Computing response times from thread patterns...")

    rows = await conn.fetch("""
        WITH thread_responses AS (
            -- For each sent email, find the most recent received email in same thread
            SELECT DISTINCT ON (sent.id)
                sent.id,
                sent.date_parsed as reply_date,
                sent.from_email,
                sent.to_emails,
                sent.subject,
                sent.thread_id,
                received.date_parsed as original_date,
                received.from_email as original_sender
            FROM emails sent
            JOIN emails received ON sent.thread_id = received.thread_id
            WHERE sent.is_sent = TRUE
              AND received.is_sent = FALSE
              AND sent.date_parsed IS NOT NULL
              AND received.date_parsed IS NOT NULL
              AND sent.date_parsed > received.date_parsed
              AND sent.thread_id IS NOT NULL
            ORDER BY sent.id, received.date_parsed DESC
        )
        SELECT
            id,
            reply_date as date_parsed,
            EXTRACT(EPOCH FROM (reply_date - original_date))::integer as response_time_seconds,
            from_email,
            to_emails,
            subject,
            original_sender
        FROM thread_responses
        WHERE EXTRACT(EPOCH FROM (reply_date - original_date)) > 0
          AND EXTRACT(EPOCH FROM (reply_date - original_date)) < 604800  -- Within 7 days
        ORDER BY reply_date
    """)

    print(f"Computed {len(rows)} response times from thread patterns")

    if len(rows) == 0:
        # Check why we got no results
        has_dates = await conn.fetchval(
            "SELECT COUNT(*) FROM emails WHERE date_parsed IS NOT NULL"
        )
        has_sent = await conn.fetchval(
            "SELECT COUNT(*) FROM emails WHERE is_sent = TRUE"
        )
        has_threads = await conn.fetchval(
            "SELECT COUNT(*) FROM emails WHERE thread_id IS NOT NULL"
        )
        print(f"\nDiagnostics:")
        print(f"  Emails with date_parsed: {has_dates}")
        print(f"  Sent emails: {has_sent}")
        print(f"  Emails with thread_id: {has_threads}")
        if has_dates == 0:
            print("\n⚠️  No emails have date_parsed populated!")
            print("   Run the enrichment pipeline to parse email dates first.")
            print("   This script requires date_parsed to compute response times.")

    return [dict(r) for r in rows]


def group_by_month(emails: list[dict]) -> dict[tuple[int, int], list[dict]]:
    """Group emails by (year, month)."""
    groups = defaultdict(list)
    for email in emails:
        dt = email['date_parsed']
        key = (dt.year, dt.month)
        groups[key].append(email)
    return groups


def analyze_period(emails: list[dict], year: int, month: int) -> Optional[TimePeriod]:
    """Analyze a month's response patterns."""
    if len(emails) < MIN_EMAILS_PER_PERIOD:
        return None

    response_times = [e['response_time_seconds'] for e in emails]

    # Get key participants (most frequent senders)
    participant_counts = defaultdict(int)
    for e in emails:
        if e['from_email']:
            participant_counts[e['from_email']] += 1
    key_participants = sorted(
        participant_counts.keys(),
        key=lambda k: participant_counts[k],
        reverse=True
    )[:5]

    # Get sample subjects
    subjects = [e['subject'] for e in emails if e['subject']][:5]

    # Calculate period boundaries
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(seconds=1)

    # Sort for median
    sorted_times = sorted(response_times)
    median = sorted_times[len(sorted_times) // 2]

    return TimePeriod(
        start=start,
        end=end,
        email_count=len(emails),
        avg_response_seconds=mean(response_times),
        median_response_seconds=median,
        key_participants=key_participants,
        sample_subjects=subjects,
    )


def find_high_engagement_periods(
    periods: list[TimePeriod],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[TimePeriod]:
    """Find periods with significantly faster response times."""
    if len(periods) < 3:
        print("Not enough periods for statistical analysis")
        return []

    # Calculate global stats
    all_avgs = [p.avg_response_seconds for p in periods]
    global_mean = mean(all_avgs)
    global_std = stdev(all_avgs) if len(all_avgs) > 1 else 0

    print(f"\nGlobal response time stats:")
    print(f"  Mean: {global_mean / 3600:.1f} hours")
    print(f"  Std dev: {global_std / 3600:.1f} hours")

    # Find periods with significantly faster response
    high_engagement = []
    for period in periods:
        z = period.z_score(global_mean, global_std)
        if z >= threshold:  # Positive z means faster than average
            high_engagement.append(period)
            print(f"  High engagement: {period.start.strftime('%Y-%m')} "
                  f"(z={z:.2f}, avg={period.avg_response_seconds / 3600:.1f}h)")

    return high_engagement


async def insert_contexts(
    conn: asyncpg.Connection,
    periods: list[TimePeriod],
    global_mean: float,
    global_std: float,
    dry_run: bool = False,
) -> int:
    """Insert discovered priority contexts into database."""
    print(f"\n{'DRY RUN - ' if dry_run else ''}Inserting {len(periods)} priority contexts...")

    inserted = 0
    for period in periods:
        z = period.z_score(global_mean, global_std)
        priority_boost = min(1.0, 0.1 + z * 0.2)  # Scale z-score to 0.1-1.0 boost

        if dry_run:
            print(f"  Would create: {period.period_name}")
            print(f"    Period: {period.start.date()} to {period.end.date()}")
            print(f"    Emails: {period.email_count}, Avg response: {period.avg_response_seconds / 3600:.1f}h")
            print(f"    Priority boost: {priority_boost:.2f}")
            inserted += 1
            continue

        # Check if similar context already exists
        existing = await conn.fetchval("""
            SELECT id FROM priority_contexts
            WHERE context_type = 'response_pattern'
            AND started_at = $1 AND ended_at = $2
        """, period.start, period.end)

        if existing:
            print(f"  Skipping (exists): {period.period_name}")
            continue

        # Insert context
        await conn.execute("""
            INSERT INTO priority_contexts (
                name, context_type, started_at, ended_at,
                priority_boost, key_participants, description
            ) VALUES (
                $1, 'response_pattern', $2, $3,
                $4, $5, $6
            )
        """,
            period.period_name,
            period.start,
            period.end,
            priority_boost,
            period.key_participants,
            f"Detected via response time analysis. {period.email_count} emails with avg response {period.avg_response_seconds / 3600:.1f}h (z-score: {z:.2f})",
        )

        print(f"  Created: {period.period_name} (boost: {priority_boost:.2f})")
        inserted += 1

    return inserted


async def main(args: argparse.Namespace):
    """Main entry point."""
    print("=" * 60)
    print("Priority Context Detection via Response Time Analysis")
    print("=" * 60)

    conn = await asyncpg.connect(DB_URL)

    try:
        # Get response data
        emails = await get_response_data(conn)

        if not emails:
            print("No emails with response times found!")
            return

        # Group by month
        monthly_groups = group_by_month(emails)
        print(f"\nAnalyzing {len(monthly_groups)} months of data...")

        # Analyze each period
        periods = []
        for (year, month), group_emails in sorted(monthly_groups.items()):
            period = analyze_period(group_emails, year, month)
            if period:
                periods.append(period)

        print(f"Found {len(periods)} months with sufficient data")

        # Find high engagement periods
        high_engagement = find_high_engagement_periods(periods, args.threshold)

        if not high_engagement:
            print("\nNo high engagement periods found meeting threshold.")
            return

        # Calculate global stats for insertion
        all_avgs = [p.avg_response_seconds for p in periods]
        global_mean = mean(all_avgs)
        global_std = stdev(all_avgs) if len(all_avgs) > 1 else 0

        # Insert contexts
        inserted = await insert_contexts(
            conn, high_engagement, global_mean, global_std, dry_run=args.dry_run
        )

        print(f"\n{'Would insert' if args.dry_run else 'Inserted'} {inserted} priority contexts")

    finally:
        await conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect priority contexts via response time analysis',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without inserting',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f'Z-score threshold for high engagement (default: {DEFAULT_THRESHOLD})',
    )

    args = parser.parse_args()
    asyncio.run(main(args))
