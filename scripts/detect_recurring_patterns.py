#!/usr/bin/env python3
"""
Detect recurring priority patterns in email data.

This script identifies calendar-based recurring patterns:
1. Q4 Planning (Oct-Dec) - Annual business planning periods
2. Annual Reviews (Dec-Jan) - Year-end review cycles
3. Board Meetings - Monthly executive activity spikes

Detection writes to priority_contexts with recurrence_pattern set.

Usage:
    python scripts/detect_recurring_patterns.py [--dry-run]
"""

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Pattern detection thresholds
MIN_EMAILS_FOR_PATTERN = 5  # Min emails to consider a period active
ACTIVITY_SPIKE_THRESHOLD = 1.5  # 50% above baseline = spike
EXEC_KEYWORDS = {
    'board', 'director', 'executive', 'ceo', 'cfo', 'coo', 'vp', 'president',
    'chairman', 'quarterly', 'budget', 'forecast', 'strategy', 'review',
    'performance', 'annual', 'planning', 'fiscal', 'q1', 'q2', 'q3', 'q4'
}
REVIEW_KEYWORDS = {
    'review', 'performance', 'annual', 'evaluation', 'assessment', 'goals',
    'objectives', 'appraisal', 'feedback', 'compensation', 'bonus', 'raise'
}


class RecurringPattern(NamedTuple):
    """A detected recurring pattern."""
    name: str
    recurrence_pattern: str
    context_type: str
    months: list[int]  # Active months (1-12)
    priority_boost: float
    keywords: list[str]
    key_participants: list[str]
    description: str


async def get_monthly_activity(conn: asyncpg.Connection) -> dict:
    """Get email activity by month across all years."""
    rows = await conn.fetch("""
        SELECT
            EXTRACT(YEAR FROM date_parsed) as year,
            EXTRACT(MONTH FROM date_parsed) as month,
            COUNT(*) as email_count,
            COUNT(DISTINCT from_email) as sender_count,
            AVG(CASE WHEN response_time_seconds IS NOT NULL
                     AND response_time_seconds > 0
                     AND response_time_seconds < 604800  -- < 1 week
                THEN response_time_seconds / 3600.0 END) as avg_response_hours,
            array_agg(DISTINCT subject) as subjects,
            array_agg(DISTINCT from_email) as senders
        FROM emails
        WHERE date_parsed IS NOT NULL
          AND NOT is_sent
        GROUP BY EXTRACT(YEAR FROM date_parsed), EXTRACT(MONTH FROM date_parsed)
        ORDER BY year, month
    """)

    # Organize by year-month
    activity = {}
    for row in rows:
        key = (int(row['year']), int(row['month']))
        activity[key] = {
            'email_count': row['email_count'],
            'sender_count': row['sender_count'],
            'avg_response_hours': float(row['avg_response_hours']) if row['avg_response_hours'] else None,
            'subjects': row['subjects'] or [],
            'senders': row['senders'] or [],
        }
    return activity


async def get_exec_activity(conn: asyncpg.Connection) -> dict:
    """Get email activity involving executive participants."""
    # Look for emails with exec-related keywords or domains
    rows = await conn.fetch("""
        SELECT
            EXTRACT(YEAR FROM date_parsed) as year,
            EXTRACT(MONTH FROM date_parsed) as month,
            EXTRACT(DAY FROM date_parsed) as day,
            COUNT(*) as email_count,
            array_agg(DISTINCT from_email) as senders,
            array_agg(DISTINCT subject) as subjects
        FROM emails
        WHERE date_parsed IS NOT NULL
          AND NOT is_sent
          AND (
              LOWER(subject) SIMILAR TO '%(board|executive|quarterly|budget|strategy|ceo|cfo)%'
              OR LOWER(from_email) SIMILAR TO '%(ceo|cfo|coo|president|director|vp)%'
          )
        GROUP BY EXTRACT(YEAR FROM date_parsed),
                 EXTRACT(MONTH FROM date_parsed),
                 EXTRACT(DAY FROM date_parsed)
        ORDER BY year, month, day
    """)

    # Organize by year-month-day
    activity = defaultdict(list)
    for row in rows:
        key = (int(row['year']), int(row['month']))
        activity[key].append({
            'day': int(row['day']),
            'email_count': row['email_count'],
            'senders': row['senders'] or [],
            'subjects': row['subjects'] or [],
        })
    return dict(activity)


def detect_q4_planning(monthly_activity: dict) -> RecurringPattern | None:
    """Detect Q4 planning patterns (Oct-Dec activity spikes)."""
    # Calculate baseline (average across all months)
    all_counts = [v['email_count'] for v in monthly_activity.values()]
    if not all_counts:
        return None
    baseline = sum(all_counts) / len(all_counts)

    # Check Q4 months (Oct=10, Nov=11, Dec=12) across all years
    q4_activity = []
    q4_keywords = set()
    q4_senders = set()

    for (year, month), data in monthly_activity.items():
        if month in [10, 11, 12]:
            q4_activity.append(data['email_count'])
            # Extract planning-related keywords from subjects
            for subj in (data['subjects'] or []):
                if subj:
                    words = subj.lower().split()
                    q4_keywords.update(w for w in words if w in EXEC_KEYWORDS)
            q4_senders.update(s for s in (data['senders'] or []) if s)

    if not q4_activity:
        return None

    # Check if Q4 has elevated activity
    q4_avg = sum(q4_activity) / len(q4_activity)
    if q4_avg < baseline * ACTIVITY_SPIKE_THRESHOLD:
        return None

    spike_ratio = q4_avg / baseline if baseline > 0 else 1.0
    priority_boost = min(2.0, 1.0 + (spike_ratio - 1) * 0.5)

    return RecurringPattern(
        name="Q4 Planning Period",
        recurrence_pattern="annual:oct-dec",
        context_type="professional",
        months=[10, 11, 12],
        priority_boost=round(priority_boost, 2),
        keywords=list(q4_keywords)[:10],
        key_participants=list(q4_senders)[:5],
        description=(
            f"Annual Q4 planning period with {spike_ratio:.1f}x baseline activity. "
            f"Detected across {len(q4_activity)} Q4 months."
        )
    )


def detect_annual_reviews(monthly_activity: dict) -> RecurringPattern | None:
    """Detect annual review patterns (Dec-Jan activity patterns)."""
    all_counts = [v['email_count'] for v in monthly_activity.values()]
    if not all_counts:
        return None
    baseline = sum(all_counts) / len(all_counts)

    # Check Dec (12) and Jan (1) across all years
    review_activity = []
    review_keywords = set()
    review_senders = set()

    for (year, month), data in monthly_activity.items():
        if month in [12, 1]:
            review_activity.append(data['email_count'])
            # Extract review-related keywords
            for subj in (data['subjects'] or []):
                if subj:
                    words = subj.lower().split()
                    review_keywords.update(w for w in words if w in REVIEW_KEYWORDS)
            review_senders.update(s for s in (data['senders'] or []) if s)

    if not review_activity:
        return None

    # Check for elevated activity or review keywords
    review_avg = sum(review_activity) / len(review_activity)
    has_spike = review_avg >= baseline * 1.2  # 20% above baseline
    has_keywords = len(review_keywords) >= 2

    if not (has_spike or has_keywords):
        return None

    spike_ratio = review_avg / baseline if baseline > 0 else 1.0
    priority_boost = min(2.0, 1.0 + (spike_ratio - 1) * 0.3 + len(review_keywords) * 0.1)

    return RecurringPattern(
        name="Annual Review Period",
        recurrence_pattern="annual:dec-jan",
        context_type="professional",
        months=[12, 1],
        priority_boost=round(priority_boost, 2),
        keywords=list(review_keywords)[:10],
        key_participants=list(review_senders)[:5],
        description=(
            f"Annual review period with {spike_ratio:.1f}x baseline activity. "
            f"Found {len(review_keywords)} review-related keywords."
        )
    )


def detect_board_meetings(exec_activity: dict) -> RecurringPattern | None:
    """Detect board meeting patterns (monthly exec spikes)."""
    if not exec_activity:
        return None

    # Look for monthly patterns - spikes on specific days
    monthly_spikes = []
    all_senders = set()
    all_subjects = []

    for (year, month), daily_data in exec_activity.items():
        if not daily_data:
            continue

        # Find the day with max exec activity in each month
        max_day = max(daily_data, key=lambda d: d['email_count'])
        if max_day['email_count'] >= 3:  # At least 3 exec emails
            monthly_spikes.append({
                'year': year,
                'month': month,
                'day': max_day['day'],
                'count': max_day['email_count'],
            })
            all_senders.update(max_day['senders'] or [])
            all_subjects.extend(max_day['subjects'] or [])

    if len(monthly_spikes) < 3:  # Need at least 3 months of pattern
        return None

    # Check for day-of-month consistency (e.g., always around 15th)
    days = [s['day'] for s in monthly_spikes]
    day_std = (sum((d - sum(days)/len(days))**2 for d in days) / len(days)) ** 0.5

    # Extract board-related keywords
    board_keywords = set()
    for subj in all_subjects:
        if subj:
            words = subj.lower().split()
            board_keywords.update(w for w in words if w in EXEC_KEYWORDS)

    avg_day = round(sum(days) / len(days))
    priority_boost = 1.8 if day_std < 7 else 1.5  # Higher boost for consistent timing

    return RecurringPattern(
        name="Board/Executive Meetings",
        recurrence_pattern=f"monthly:~day-{avg_day}",
        context_type="professional",
        months=list(range(1, 13)),  # All months
        priority_boost=round(priority_boost, 2),
        keywords=list(board_keywords)[:10],
        key_participants=list(all_senders)[:5],
        description=(
            f"Monthly executive activity pattern. "
            f"Spikes detected around day {avg_day} (±{day_std:.0f} days) "
            f"across {len(monthly_spikes)} months."
        )
    )


async def ensure_recurrence_column(conn: asyncpg.Connection) -> None:
    """Ensure recurrence_pattern column exists in priority_contexts."""
    # Check if column exists
    exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'priority_contexts'
            AND column_name = 'recurrence_pattern'
        )
    """)

    if not exists:
        print("Adding recurrence_pattern column to priority_contexts...")
        await conn.execute("""
            ALTER TABLE priority_contexts
            ADD COLUMN IF NOT EXISTS recurrence_pattern TEXT
        """)
        # Also ensure other columns exist that we need
        await conn.execute("""
            ALTER TABLE priority_contexts
            ADD COLUMN IF NOT EXISTS name TEXT,
            ADD COLUMN IF NOT EXISTS context_type TEXT,
            ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS priority_boost FLOAT,
            ADD COLUMN IF NOT EXISTS description TEXT
        """)


async def save_pattern(conn: asyncpg.Connection, pattern: RecurringPattern) -> int:
    """Save a recurring pattern to priority_contexts."""
    from datetime import date

    # For recurring patterns, use a representative week_start based on active months
    # Use the first day of the first active month in current year as reference
    first_month = min(pattern.months)
    current_year = datetime.now().year
    week_start = date(current_year, first_month, 1)

    context_id = await conn.fetchval("""
        INSERT INTO priority_contexts (
            week_start, name, context_type, priority_boost,
            keywords, key_participants, description,
            recurrence_pattern, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        ON CONFLICT DO NOTHING
        RETURNING id
    """,
        week_start,
        pattern.name,
        pattern.context_type,
        pattern.priority_boost,
        pattern.keywords,
        pattern.key_participants,
        pattern.description,
        pattern.recurrence_pattern,
    )
    return context_id


async def main(dry_run: bool = False):
    """Main entry point."""
    print("Detect Recurring Priority Patterns")
    print("=" * 40)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Ensure schema is ready
        if not dry_run:
            await ensure_recurrence_column(conn)

        # Get activity data
        print("\nAnalyzing email activity patterns...")
        monthly_activity = await get_monthly_activity(conn)
        print(f"  Found {len(monthly_activity)} month-year combinations")

        exec_activity = await get_exec_activity(conn)
        print(f"  Found exec activity in {len(exec_activity)} months")

        # Detect patterns
        patterns = []

        print("\nDetecting Q4 planning patterns...")
        q4_pattern = detect_q4_planning(monthly_activity)
        if q4_pattern:
            patterns.append(q4_pattern)
            print(f"  ✓ Found: {q4_pattern.name}")
        else:
            print("  - No significant Q4 planning pattern detected")

        print("\nDetecting annual review patterns...")
        review_pattern = detect_annual_reviews(monthly_activity)
        if review_pattern:
            patterns.append(review_pattern)
            print(f"  ✓ Found: {review_pattern.name}")
        else:
            print("  - No significant annual review pattern detected")

        print("\nDetecting board meeting patterns...")
        board_pattern = detect_board_meetings(exec_activity)
        if board_pattern:
            patterns.append(board_pattern)
            print(f"  ✓ Found: {board_pattern.name}")
        else:
            print("  - No significant board meeting pattern detected")

        if not patterns:
            print("\nNo recurring patterns detected.")
            return

        # Display patterns
        print("\n" + "=" * 60)
        print("Detected Recurring Patterns:")
        print("=" * 60)

        for p in patterns:
            print(f"\n{p.name}")
            print(f"  Recurrence: {p.recurrence_pattern}")
            print(f"  Priority Boost: {p.priority_boost}x")
            if p.keywords:
                print(f"  Keywords: {', '.join(p.keywords[:5])}")
            if p.key_participants:
                print(f"  Key Participants: {len(p.key_participants)} people")
            print(f"  {p.description}")

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Save patterns
        print("\nSaving patterns to priority_contexts...")
        for p in patterns:
            ctx_id = await save_pattern(conn, p)
            if ctx_id:
                print(f"  Created: {p.name} (id={ctx_id})")
            else:
                print(f"  Skipped: {p.name} (may already exist)")

        # Verification
        print("\n=== Verification ===")
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM priority_contexts
            WHERE recurrence_pattern IS NOT NULL
        """)
        print(f"Total recurring patterns in DB: {count}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect recurring priority patterns in email data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be detected without making changes",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run))
