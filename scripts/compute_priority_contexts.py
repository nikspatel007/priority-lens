#!/usr/bin/env python3
"""
Compute priority_contexts from email patterns.

This script orchestrates all priority context detection methods:
1. Response Time Analysis - Fast-response periods indicating high engagement
2. Recurring Patterns - Calendar-based patterns (Q4, annual reviews, board meetings)
3. Participant Bursts - Domain activity spikes (job search, real estate, etc.)

The detected contexts are stored in the priority_contexts table and can be
used to boost email priority during active engagement periods.

Usage:
    python scripts/compute_priority_contexts.py [--dry-run] [--force]

Options:
    --dry-run   Show what would be detected without making changes
    --force     Clear existing contexts before computing new ones
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, date
from statistics import mean, stdev
from typing import NamedTuple
import re

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# ============================================
# Response Time Analysis Thresholds
# ============================================
FAST_RESPONSE_THRESHOLD = 0.5  # Response < 50% of average = fast
MIN_EMAILS_PER_WEEK = 3  # Minimum emails to consider a week
MAX_RESPONSE_SECONDS = 86400 * 7  # Ignore responses > 1 week (outliers)
MIN_CONTEXT_WEEKS = 2  # Minimum weeks to form a context
MAX_GAP_WEEKS = 2  # Maximum gap between weeks to still group together
PRIORITY_BOOST_BASE = 1.5  # Base priority boost for fast-response contexts

# ============================================
# Recurring Pattern Thresholds
# ============================================
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

# ============================================
# Participant Burst Thresholds
# ============================================
MIN_EMAILS_FOR_DOMAIN = 5  # Minimum emails from domain to consider
SPIKE_THRESHOLD = 2.0  # Activity > 2x average = spike
MIN_SPIKE_MONTHS = 1  # Minimum months to form a context
MAX_GAP_MONTHS = 2  # Maximum gap between months to group together

IGNORE_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
    'googlemail.com', 'aol.com', 'icloud.com', 'me.com',
    'mail.com', 'protonmail.com', 'pm.me',
}

DOMAIN_CATEGORIES = {
    'linkedin.com': 'Professional Networking',
    'indeed.com': 'Job Search',
    'glassdoor.com': 'Job Search',
    'lever.co': 'Job Search',
    'greenhouse.io': 'Job Search',
    'workday.com': 'Job Search',
    'amazon.com': 'Shopping',
    'ebay.com': 'Shopping',
    'zillow.com': 'Real Estate',
    'redfin.com': 'Real Estate',
    'realtor.com': 'Real Estate',
    'github.com': 'Development',
    'gitlab.com': 'Development',
    'stripe.com': 'Payments',
    'paypal.com': 'Payments',
}

# Stop words for keyword extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'your', 'my', 'our', 'their', 'this', 'that', 'these', 'those',
    're', 'fwd', 'fw', 'you', 'i', 'we', 'they', 'it', 'he', 'she',
    'new', 'please', 'thanks', 'thank', 'hi', 'hello', 'hey',
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


# ============================================
# Utility Functions
# ============================================

def extract_keywords(subjects: list[str], top_n: int = 5) -> list[str]:
    """Extract common keywords from email subjects."""
    if not subjects:
        return []

    word_counts = Counter()
    for subject in subjects:
        if not subject:
            continue
        words = re.findall(r'\b[a-zA-Z]{3,}\b', subject.lower())
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        word_counts.update(words)

    return [word for word, _ in word_counts.most_common(top_n)]


def extract_key_participants(senders: list[str], top_n: int = 5) -> list[str]:
    """Extract most frequent senders, filtering generic domains."""
    if not senders:
        return []

    generic_domains = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'googlemail.com', 'aol.com', 'icloud.com', 'me.com',
    }

    filtered = []
    for sender in senders:
        if not sender or '@' not in sender:
            continue
        domain = sender.split('@')[1].lower()
        if domain not in generic_domains:
            filtered.append(sender)

    if not filtered:
        filtered = [s for s in senders if s]

    return [p[0] for p in Counter(filtered).most_common(top_n)]


# ============================================
# Response Time Analysis
# ============================================

async def get_weekly_response_times(conn: asyncpg.Connection) -> list[dict]:
    """Get weekly average response times."""
    rows = await conn.fetch("""
        WITH weekly_stats AS (
            SELECT
                date_trunc('week', date_parsed) as week,
                AVG(response_time_seconds) / 3600 as avg_hours,
                COUNT(*) as email_count,
                array_agg(DISTINCT from_email) as senders,
                array_agg(DISTINCT subject) as subjects
            FROM emails
            WHERE response_time_seconds IS NOT NULL
              AND response_time_seconds > 0
              AND response_time_seconds < $1
              AND NOT is_sent
              AND date_parsed IS NOT NULL
            GROUP BY week
            HAVING COUNT(*) >= $2
        )
        SELECT * FROM weekly_stats ORDER BY week
    """, MAX_RESPONSE_SECONDS, MIN_EMAILS_PER_WEEK)
    return [dict(row) for row in rows]


def group_consecutive_weeks(fast_weeks: list[dict]) -> list[list[dict]]:
    """Group consecutive fast weeks into contexts."""
    if not fast_weeks:
        return []

    groups = []
    current_group = [fast_weeks[0]]

    for week in fast_weeks[1:]:
        prev_week = current_group[-1]['week']
        curr_week = week['week']
        gap = (curr_week - prev_week).days // 7

        if gap <= MAX_GAP_WEEKS:
            current_group.append(week)
        else:
            if len(current_group) >= MIN_CONTEXT_WEEKS:
                groups.append(current_group)
            current_group = [week]

    if len(current_group) >= MIN_CONTEXT_WEEKS:
        groups.append(current_group)

    return groups


def create_response_time_context(group: list[dict], overall_avg: float) -> dict:
    """Create a priority context from a group of fast weeks."""
    all_senders = []
    all_subjects = []
    total_emails = 0
    total_hours = 0

    for week in group:
        all_senders.extend(week['senders'] or [])
        all_subjects.extend(week['subjects'] or [])
        total_emails += week['email_count']
        total_hours += week['avg_hours'] * week['email_count']

    avg_response = total_hours / total_emails if total_emails > 0 else 0
    speed_ratio = avg_response / overall_avg if overall_avg > 0 else 1
    priority_boost = PRIORITY_BOOST_BASE + (1 - speed_ratio)

    keywords = extract_keywords(all_subjects)
    key_participants = extract_key_participants(all_senders)

    start_date = group[0]['week']
    end_date = group[-1]['week'] + timedelta(days=6)

    if keywords:
        name = f"High Priority: {keywords[0].title()}"
        if len(keywords) > 1:
            name += f" / {keywords[1].title()}"
    else:
        name = f"Fast Response Period"

    name += f" ({start_date.strftime('%b %Y')})"

    return {
        'week_start': start_date.date() if hasattr(start_date, 'date') else start_date,
        'name': name[:100],
        'context_type': 'response_time',
        'priority_boost': round(priority_boost, 2),
        'keywords': keywords,
        'key_participants': key_participants[:5],
        'description': (
            f"Period of heightened email engagement. "
            f"Avg response: {avg_response:.1f}h (vs {overall_avg:.1f}h overall). "
            f"{total_emails} emails over {len(group)} weeks."
        ),
        'email_count': total_emails,
        'avg_response_hours': avg_response,
        'baseline_hours': overall_avg,
        'deviation_factor': 1 - speed_ratio if speed_ratio < 1 else 0,
    }


async def detect_response_time_contexts(conn: asyncpg.Connection) -> list[dict]:
    """Detect priority contexts via response time analysis."""
    print("\n--- Response Time Analysis ---")

    weeks = await get_weekly_response_times(conn)
    print(f"  Found {len(weeks)} weeks with sufficient data")

    if not weeks:
        return []

    # Calculate overall average
    for w in weeks:
        w['avg_hours'] = float(w['avg_hours']) if w['avg_hours'] else 0.0
    total_hours = sum(w['avg_hours'] * w['email_count'] for w in weeks)
    total_emails = sum(w['email_count'] for w in weeks)
    overall_avg = total_hours / total_emails if total_emails > 0 else 0
    print(f"  Overall average response time: {overall_avg:.1f} hours")

    # Find fast-response weeks
    threshold = overall_avg * FAST_RESPONSE_THRESHOLD
    fast_weeks = [w for w in weeks if w['avg_hours'] < threshold]
    print(f"  Found {len(fast_weeks)} fast-response weeks (< {threshold:.1f}h)")

    if not fast_weeks:
        return []

    # Group consecutive weeks
    groups = group_consecutive_weeks(fast_weeks)
    print(f"  Grouped into {len(groups)} distinct contexts")

    contexts = [create_response_time_context(g, overall_avg) for g in groups]
    return contexts


# ============================================
# Recurring Pattern Detection
# ============================================

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
                     AND response_time_seconds < 604800
                THEN response_time_seconds / 3600.0 END) as avg_response_hours,
            array_agg(DISTINCT subject) as subjects,
            array_agg(DISTINCT from_email) as senders
        FROM emails
        WHERE date_parsed IS NOT NULL
          AND NOT is_sent
        GROUP BY EXTRACT(YEAR FROM date_parsed), EXTRACT(MONTH FROM date_parsed)
        ORDER BY year, month
    """)

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
    all_counts = [v['email_count'] for v in monthly_activity.values()]
    if not all_counts:
        return None
    baseline = sum(all_counts) / len(all_counts)

    q4_activity = []
    q4_keywords = set()
    q4_senders = set()

    for (year, month), data in monthly_activity.items():
        if month in [10, 11, 12]:
            q4_activity.append(data['email_count'])
            for subj in (data['subjects'] or []):
                if subj:
                    words = subj.lower().split()
                    q4_keywords.update(w for w in words if w in EXEC_KEYWORDS)
            q4_senders.update(s for s in (data['senders'] or []) if s)

    if not q4_activity:
        return None

    q4_avg = sum(q4_activity) / len(q4_activity)
    if q4_avg < baseline * ACTIVITY_SPIKE_THRESHOLD:
        return None

    spike_ratio = q4_avg / baseline if baseline > 0 else 1.0
    priority_boost = min(2.0, 1.0 + (spike_ratio - 1) * 0.5)

    return RecurringPattern(
        name="Q4 Planning Period",
        recurrence_pattern="annual:oct-dec",
        context_type="recurring",
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

    review_activity = []
    review_keywords = set()
    review_senders = set()

    for (year, month), data in monthly_activity.items():
        if month in [12, 1]:
            review_activity.append(data['email_count'])
            for subj in (data['subjects'] or []):
                if subj:
                    words = subj.lower().split()
                    review_keywords.update(w for w in words if w in REVIEW_KEYWORDS)
            review_senders.update(s for s in (data['senders'] or []) if s)

    if not review_activity:
        return None

    review_avg = sum(review_activity) / len(review_activity)
    has_spike = review_avg >= baseline * 1.2
    has_keywords = len(review_keywords) >= 2

    if not (has_spike or has_keywords):
        return None

    spike_ratio = review_avg / baseline if baseline > 0 else 1.0
    priority_boost = min(2.0, 1.0 + (spike_ratio - 1) * 0.3 + len(review_keywords) * 0.1)

    return RecurringPattern(
        name="Annual Review Period",
        recurrence_pattern="annual:dec-jan",
        context_type="recurring",
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

    monthly_spikes = []
    all_senders = set()
    all_subjects = []

    for (year, month), daily_data in exec_activity.items():
        if not daily_data:
            continue
        max_day = max(daily_data, key=lambda d: d['email_count'])
        if max_day['email_count'] >= 3:
            monthly_spikes.append({
                'year': year,
                'month': month,
                'day': max_day['day'],
                'count': max_day['email_count'],
            })
            all_senders.update(max_day['senders'] or [])
            all_subjects.extend(max_day['subjects'] or [])

    if len(monthly_spikes) < 3:
        return None

    days = [s['day'] for s in monthly_spikes]
    day_std = (sum((d - sum(days)/len(days))**2 for d in days) / len(days)) ** 0.5

    board_keywords = set()
    for subj in all_subjects:
        if subj:
            words = subj.lower().split()
            board_keywords.update(w for w in words if w in EXEC_KEYWORDS)

    avg_day = round(sum(days) / len(days))
    priority_boost = 1.8 if day_std < 7 else 1.5

    return RecurringPattern(
        name="Board/Executive Meetings",
        recurrence_pattern=f"monthly:~day-{avg_day}",
        context_type="recurring",
        months=list(range(1, 13)),
        priority_boost=round(priority_boost, 2),
        keywords=list(board_keywords)[:10],
        key_participants=list(all_senders)[:5],
        description=(
            f"Monthly executive activity pattern. "
            f"Spikes detected around day {avg_day} (±{day_std:.0f} days) "
            f"across {len(monthly_spikes)} months."
        )
    )


async def detect_recurring_patterns(conn: asyncpg.Connection) -> list[dict]:
    """Detect recurring priority patterns."""
    print("\n--- Recurring Pattern Detection ---")

    monthly_activity = await get_monthly_activity(conn)
    print(f"  Found {len(monthly_activity)} month-year combinations")

    exec_activity = await get_exec_activity(conn)
    print(f"  Found exec activity in {len(exec_activity)} months")

    patterns = []
    detected = []

    q4_pattern = detect_q4_planning(monthly_activity)
    if q4_pattern:
        patterns.append(q4_pattern)
        detected.append("Q4 Planning")

    review_pattern = detect_annual_reviews(monthly_activity)
    if review_pattern:
        patterns.append(review_pattern)
        detected.append("Annual Reviews")

    board_pattern = detect_board_meetings(exec_activity)
    if board_pattern:
        patterns.append(board_pattern)
        detected.append("Board Meetings")

    if detected:
        print(f"  Detected: {', '.join(detected)}")
    else:
        print("  No recurring patterns detected")

    # Convert patterns to context dicts
    contexts = []
    current_year = datetime.now().year
    for p in patterns:
        first_month = min(p.months)
        week_start = date(current_year, first_month, 1)
        contexts.append({
            'week_start': week_start,
            'name': p.name,
            'context_type': p.context_type,
            'priority_boost': p.priority_boost,
            'keywords': p.keywords,
            'key_participants': p.key_participants,
            'description': p.description,
            'recurrence_pattern': p.recurrence_pattern,
            'email_count': None,
            'avg_response_hours': None,
            'baseline_hours': None,
            'deviation_factor': None,
        })

    return contexts


# ============================================
# Participant Burst Detection
# ============================================

async def get_monthly_domain_activity(conn: asyncpg.Connection) -> list[dict]:
    """Get email counts grouped by month and sender domain."""
    rows = await conn.fetch("""
        SELECT
            date_trunc('month', date_parsed) as month,
            LOWER(SPLIT_PART(from_email, '@', 2)) as domain,
            COUNT(*) as email_count,
            array_agg(DISTINCT subject) as sample_subjects
        FROM emails
        WHERE date_parsed IS NOT NULL
          AND from_email IS NOT NULL
          AND NOT is_sent
        GROUP BY month, domain
        HAVING COUNT(*) >= 1
        ORDER BY month, domain
    """)
    return [dict(row) for row in rows]


def calculate_domain_baselines(activity: list[dict]) -> dict[str, dict]:
    """Calculate baseline activity stats per domain."""
    domain_months = defaultdict(list)

    for row in activity:
        domain = row['domain']
        if domain not in IGNORE_DOMAINS:
            domain_months[domain].append(row['email_count'])

    baselines = {}
    for domain, counts in domain_months.items():
        if len(counts) >= 2:
            baselines[domain] = {
                'mean': mean(counts),
                'stdev': stdev(counts) if len(counts) > 1 else 0,
                'total_months': len(counts),
                'total_emails': sum(counts),
            }
        elif len(counts) == 1 and counts[0] >= MIN_EMAILS_FOR_DOMAIN:
            baselines[domain] = {
                'mean': counts[0],
                'stdev': 0,
                'total_months': 1,
                'total_emails': counts[0],
            }

    return baselines


def detect_domain_spikes(activity: list[dict], baselines: dict) -> list[dict]:
    """Find months where domain activity spikes above baseline."""
    spikes = []

    for row in activity:
        domain = row['domain']
        if domain in IGNORE_DOMAINS or domain not in baselines:
            continue

        baseline = baselines[domain]
        count = row['email_count']

        if count >= baseline['mean'] * SPIKE_THRESHOLD and count >= MIN_EMAILS_FOR_DOMAIN:
            deviation = (count - baseline['mean']) / baseline['mean'] if baseline['mean'] > 0 else 0
            spikes.append({
                'month': row['month'],
                'domain': domain,
                'email_count': count,
                'baseline_mean': baseline['mean'],
                'deviation_factor': deviation,
                'sample_subjects': row['sample_subjects'][:5] if row['sample_subjects'] else [],
            })

    return sorted(spikes, key=lambda x: (x['month'], -x['deviation_factor']))


def group_domain_spikes_into_contexts(spikes: list[dict]) -> list[dict]:
    """Group consecutive spike months by domain into contexts."""
    if not spikes:
        return []

    domain_spikes = defaultdict(list)
    for spike in spikes:
        domain_spikes[spike['domain']].append(spike)

    contexts = []
    for domain, domain_spike_list in domain_spikes.items():
        sorted_spikes = sorted(domain_spike_list, key=lambda x: x['month'])

        groups = []
        current_group = [sorted_spikes[0]]

        for spike in sorted_spikes[1:]:
            prev_month = current_group[-1]['month']
            curr_month = spike['month']
            gap_months = (curr_month.year - prev_month.year) * 12 + (curr_month.month - prev_month.month)

            if gap_months <= MAX_GAP_MONTHS:
                current_group.append(spike)
            else:
                if len(current_group) >= MIN_SPIKE_MONTHS:
                    groups.append(current_group)
                current_group = [spike]

        if len(current_group) >= MIN_SPIKE_MONTHS:
            groups.append(current_group)

        for group in groups:
            ctx = create_burst_context(domain, group)
            if ctx:
                contexts.append(ctx)

    return sorted(contexts, key=lambda x: x['week_start'])


def create_burst_context(domain: str, group: list[dict]) -> dict:
    """Create a priority context from a group of spike months."""
    total_emails = sum(s['email_count'] for s in group)
    avg_deviation = mean(s['deviation_factor'] for s in group)
    avg_baseline = mean(s['baseline_mean'] for s in group)

    all_subjects = []
    for s in group:
        all_subjects.extend(s.get('sample_subjects', []))

    keywords = extract_keywords(all_subjects)
    category = DOMAIN_CATEGORIES.get(domain, 'General')

    start_month = min(s['month'] for s in group)
    end_month = max(s['month'] for s in group)

    return {
        'week_start': start_month.date() if hasattr(start_month, 'date') else start_month,
        'name': f"{category}: {domain}",
        'context_type': 'participant_burst',
        'priority_boost': round(1.0 + min(avg_deviation * 0.3, 1.0), 2),
        'keywords': keywords + [domain],
        'key_participants': [domain],
        'description': (
            f"Activity burst from {domain} ({category}). "
            f"{total_emails} emails, {avg_deviation:.1f}x baseline "
            f"from {start_month.strftime('%Y-%m')} to {end_month.strftime('%Y-%m')}."
        ),
        'email_count': total_emails,
        'avg_response_hours': None,
        'baseline_hours': avg_baseline,
        'deviation_factor': avg_deviation,
    }


async def detect_participant_bursts(conn: asyncpg.Connection) -> list[dict]:
    """Detect priority contexts via participant bursts."""
    print("\n--- Participant Burst Detection ---")

    activity = await get_monthly_domain_activity(conn)
    print(f"  Found {len(activity)} (month, domain) combinations")

    if not activity:
        return []

    baselines = calculate_domain_baselines(activity)
    print(f"  Computed baselines for {len(baselines)} domains")

    spikes = detect_domain_spikes(activity, baselines)
    print(f"  Found {len(spikes)} spike events")

    if not spikes:
        return []

    contexts = group_domain_spikes_into_contexts(spikes)
    print(f"  Created {len(contexts)} burst contexts")

    return contexts


# ============================================
# Database Operations
# ============================================

async def ensure_schema(conn: asyncpg.Connection) -> None:
    """Ensure priority_contexts table has required columns."""
    # Check for recurrence_pattern column
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

    # Check for name column
    exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'priority_contexts'
            AND column_name = 'name'
        )
    """)

    if not exists:
        print("Adding name column to priority_contexts...")
        await conn.execute("""
            ALTER TABLE priority_contexts
            ADD COLUMN IF NOT EXISTS name TEXT,
            ADD COLUMN IF NOT EXISTS context_type TEXT,
            ADD COLUMN IF NOT EXISTS priority_boost FLOAT,
            ADD COLUMN IF NOT EXISTS description TEXT
        """)


async def clear_contexts(conn: asyncpg.Connection) -> int:
    """Clear all existing priority contexts."""
    result = await conn.execute("DELETE FROM priority_contexts")
    count = int(result.split()[-1]) if result else 0
    return count


async def save_context(conn: asyncpg.Connection, context: dict) -> int:
    """Save a priority context to the database."""
    context_id = await conn.fetchval("""
        INSERT INTO priority_contexts (
            week_start, name, context_type, priority_boost,
            email_count, avg_response_hours, baseline_hours, deviation_factor,
            keywords, key_participants, description,
            recurrence_pattern, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
        RETURNING id
    """,
        context.get('week_start'),
        context.get('name'),
        context.get('context_type'),
        context.get('priority_boost'),
        context.get('email_count'),
        context.get('avg_response_hours'),
        context.get('baseline_hours'),
        context.get('deviation_factor'),
        context.get('keywords'),
        context.get('key_participants'),
        context.get('description'),
        context.get('recurrence_pattern'),
    )
    return context_id


# ============================================
# Main Entry Point
# ============================================

async def main(dry_run: bool = False, force: bool = False):
    """Main entry point."""
    print("=" * 60)
    print("Compute Priority Contexts from Email Patterns")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN - no changes will be made]\n")

    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)

    try:
        # Ensure schema is ready
        if not dry_run:
            await ensure_schema(conn)

        # Clear existing contexts if force flag is set
        if force and not dry_run:
            cleared = await clear_contexts(conn)
            print(f"\nCleared {cleared} existing priority contexts")

        # Collect all contexts from all detection methods
        all_contexts = []

        # 1. Response Time Analysis
        response_contexts = await detect_response_time_contexts(conn)
        all_contexts.extend(response_contexts)

        # 2. Recurring Pattern Detection
        recurring_contexts = await detect_recurring_patterns(conn)
        all_contexts.extend(recurring_contexts)

        # 3. Participant Burst Detection
        burst_contexts = await detect_participant_bursts(conn)
        all_contexts.extend(burst_contexts)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Response Time Contexts: {len(response_contexts)}")
        print(f"  Recurring Patterns: {len(recurring_contexts)}")
        print(f"  Participant Bursts: {len(burst_contexts)}")
        print(f"  TOTAL: {len(all_contexts)}")

        if not all_contexts:
            print("\nNo priority contexts detected.")
            return

        # Display discovered contexts
        print("\n" + "=" * 60)
        print("Discovered Priority Contexts:")
        print("=" * 60)

        for ctx in all_contexts:
            print(f"\n{ctx.get('name', 'Unknown')}")
            print(f"  Type: {ctx.get('context_type', 'unknown')}")
            print(f"  Week Start: {ctx.get('week_start')}")
            if ctx.get('priority_boost'):
                print(f"  Priority Boost: {ctx['priority_boost']}x")
            if ctx.get('keywords'):
                print(f"  Keywords: {', '.join(ctx['keywords'][:5])}")
            if ctx.get('recurrence_pattern'):
                print(f"  Recurrence: {ctx['recurrence_pattern']}")

        if dry_run:
            print("\n[DRY RUN - stopping here]")
            return

        # Save contexts to database
        print("\nSaving contexts to database...")
        saved_count = 0
        for ctx in all_contexts:
            ctx_id = await save_context(conn, ctx)
            if ctx_id:
                saved_count += 1
                print(f"  Created: {ctx.get('name', 'Unknown')[:50]}... (id={ctx_id})")

        # Verification
        print("\n=== Verification ===")
        total = await conn.fetchval("SELECT COUNT(*) FROM priority_contexts")
        by_type = await conn.fetch("""
            SELECT context_type, COUNT(*) as cnt
            FROM priority_contexts
            GROUP BY context_type
            ORDER BY cnt DESC
        """)

        print(f"Total priority contexts in DB: {total}")
        print("By type:")
        for row in by_type:
            print(f"  {row['context_type'] or 'unknown'}: {row['cnt']}")

    finally:
        await conn.close()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute priority contexts from email patterns"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be detected without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing contexts before computing new ones",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, force=args.force))
