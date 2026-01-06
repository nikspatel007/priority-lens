#!/usr/bin/env python3
"""Populate tasks table from email_llm_features.tasks JSON.

Extracts tasks from the JSONB tasks column in email_llm_features and
populates the tasks table with normalized data.

Each task in the JSON has:
- description: task description
- deadline: optional deadline text
- assignee: "user" | "other" | null
- task_type: "review" | "send" | "schedule" | "decision" | "create" | "followup"
- urgency: 0.0-1.0

Usage:
    python scripts/populate_tasks.py
    python scripts/populate_tasks.py --db-url postgresql://user:pass@host:port/db
    python scripts/populate_tasks.py --stats-only
"""

import argparse
import hashlib
import psycopg2
from psycopg2.extras import execute_values


# Map LLM-extracted task types to schema-valid values
TASK_TYPE_MAP = {
    "review": "review",
    "send": "send",
    "schedule": "schedule",
    "decision": "decision",
    "create": "create",
    "followup": "follow_up",
    "follow_up": "follow_up",
    "research": "research",
    "other": "other",
}

# Valid task types per schema constraint
VALID_TASK_TYPES = frozenset([
    "review", "send", "schedule", "decision",
    "research", "create", "follow_up", "other"
])


def get_connection(db_url: str):
    """Connect to PostgreSQL database."""
    return psycopg2.connect(db_url)


def generate_task_id(email_id: str, index: int, description: str) -> str:
    """Generate a unique task ID.

    Uses a hash of email_id + description to be deterministic across runs.
    """
    content = f"{email_id}:{index}:{description}"
    hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"task-{hash_suffix}"


def normalize_task_type(raw_type: str | None) -> str:
    """Normalize task type to schema-valid value."""
    if not raw_type:
        return "other"
    normalized = TASK_TYPE_MAP.get(raw_type.lower().strip(), "other")
    return normalized if normalized in VALID_TASK_TYPES else "other"


def normalize_urgency(urgency: float | None) -> float:
    """Clamp urgency to valid range [0, 1]."""
    if urgency is None:
        return 0.0
    return max(0.0, min(1.0, float(urgency)))


def populate_tasks(conn) -> dict:
    """Extract tasks from email_llm_features and populate tasks table.

    Returns statistics dict.
    """
    cur = conn.cursor()

    stats = {
        "emails_with_tasks": 0,
        "tasks_inserted": 0,
        "tasks_skipped": 0,
        "tasks_updated": 0,
    }

    # Check if email_llm_features table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'email_llm_features'
        )
    """)
    if not cur.fetchone()[0]:
        print("Error: email_llm_features table does not exist.")
        print("Run extract_llm_features.py first to populate it.")
        return stats

    # Check if tasks table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'tasks'
        )
    """)
    if not cur.fetchone()[0]:
        print("Error: tasks table does not exist.")
        print("Run alembic upgrade head or create_schema.sql first.")
        return stats

    # Get existing task count
    cur.execute("SELECT COUNT(*) FROM tasks")
    existing_count = cur.fetchone()[0]
    print(f"Existing tasks in table: {existing_count}")

    # Clear existing tasks for fresh population
    if existing_count > 0:
        print("Clearing existing tasks for fresh population...")
        cur.execute("TRUNCATE tasks RESTART IDENTITY CASCADE")

    # Query email_llm_features with tasks, joined to emails for FK
    print("Querying email_llm_features for tasks...")
    cur.execute("""
        SELECT
            f.email_id as message_id,
            e.id as email_pk,
            f.tasks
        FROM email_llm_features f
        JOIN emails e ON e.message_id = f.email_id
        WHERE f.tasks IS NOT NULL
          AND jsonb_array_length(f.tasks) > 0
          AND f.parse_success = true
    """)

    rows = cur.fetchall()
    print(f"Found {len(rows)} emails with tasks")

    # Collect all tasks for batch insert
    task_records = []

    for message_id, email_pk, tasks_json in rows:
        if not tasks_json:
            continue

        stats["emails_with_tasks"] += 1

        for idx, task in enumerate(tasks_json):
            if not isinstance(task, dict):
                stats["tasks_skipped"] += 1
                continue

            description = task.get("description", "").strip()
            if not description:
                stats["tasks_skipped"] += 1
                continue

            task_id = generate_task_id(message_id, idx, description)
            task_type = normalize_task_type(task.get("task_type"))
            urgency_score = normalize_urgency(task.get("urgency"))
            deadline_text = task.get("deadline")
            assignee = task.get("assignee")

            # Map assignee to hint text
            assignee_hint = None
            if assignee == "user":
                assignee_hint = "assigned to user"
            elif assignee == "other":
                assignee_hint = "assigned to other"

            task_records.append((
                task_id,
                email_pk,
                description[:1000],  # Truncate to schema limit
                deadline_text,
                assignee_hint,
                "unknown",  # complexity not extracted by LLM
                task_type,
                urgency_score,
                description[:500] if description else None,  # source_text
            ))

    if not task_records:
        print("No valid tasks found to insert")
        return stats

    print(f"Inserting {len(task_records)} tasks...")

    # Batch insert
    execute_values(
        cur,
        """
        INSERT INTO tasks (
            task_id, email_id, description, deadline_text, assignee_hint,
            complexity, task_type, urgency_score, source_text
        ) VALUES %s
        ON CONFLICT (task_id) DO UPDATE SET
            description = EXCLUDED.description,
            deadline_text = EXCLUDED.deadline_text,
            assignee_hint = EXCLUDED.assignee_hint,
            task_type = EXCLUDED.task_type,
            urgency_score = EXCLUDED.urgency_score
        """,
        task_records,
        page_size=1000
    )

    stats["tasks_inserted"] = len(task_records)

    conn.commit()
    cur.close()

    return stats


def print_stats(conn):
    """Print statistics about populated tasks."""
    cur = conn.cursor()

    print("\n=== Task Population Stats ===")

    cur.execute("SELECT COUNT(*) FROM tasks")
    total = cur.fetchone()[0]
    print(f"Total tasks: {total}")

    if total == 0:
        cur.close()
        return

    cur.execute("SELECT COUNT(DISTINCT email_id) FROM tasks")
    print(f"Emails with tasks: {cur.fetchone()[0]}")

    # Task type distribution
    cur.execute("""
        SELECT task_type, COUNT(*) as cnt
        FROM tasks
        GROUP BY task_type
        ORDER BY cnt DESC
    """)
    print("\nTask types:")
    for task_type, count in cur.fetchall():
        print(f"  {task_type}: {count}")

    # Urgency distribution
    cur.execute("""
        SELECT
            CASE
                WHEN urgency_score < 0.3 THEN 'low (0-0.3)'
                WHEN urgency_score < 0.6 THEN 'medium (0.3-0.6)'
                ELSE 'high (0.6-1.0)'
            END as urgency_bucket,
            COUNT(*) as cnt
        FROM tasks
        GROUP BY 1
        ORDER BY 1
    """)
    print("\nUrgency distribution:")
    for bucket, count in cur.fetchall():
        print(f"  {bucket}: {count}")

    # Tasks with deadlines
    cur.execute("""
        SELECT COUNT(*) FROM tasks WHERE deadline_text IS NOT NULL
    """)
    print(f"\nTasks with deadline text: {cur.fetchone()[0]}")

    # Assignee distribution
    cur.execute("""
        SELECT
            COALESCE(assignee_hint, 'unassigned') as assignee,
            COUNT(*) as cnt
        FROM tasks
        GROUP BY 1
        ORDER BY cnt DESC
    """)
    print("\nAssignee distribution:")
    for assignee, count in cur.fetchall():
        print(f"  {assignee}: {count}")

    # Sample tasks
    cur.execute("""
        SELECT task_id, description, task_type, urgency_score
        FROM tasks
        ORDER BY urgency_score DESC
        LIMIT 5
    """)
    print("\n=== Top 5 urgent tasks ===")
    for task_id, desc, task_type, urgency in cur.fetchall():
        desc_preview = desc[:60] + "..." if len(desc) > 60 else desc
        print(f"  [{urgency:.2f}] {task_type}: {desc_preview}")

    cur.close()


def main():
    parser = argparse.ArgumentParser(
        description="Populate tasks table from email_llm_features.tasks JSON"
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://postgres:postgres@localhost:5433/rl_emails",
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print stats, do not populate"
    )

    args = parser.parse_args()

    print("Populate Tasks Table")
    print("=" * 40)
    print(f"Connecting to database...")

    try:
        conn = get_connection(args.db_url)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return 1

    try:
        if not args.stats_only:
            stats = populate_tasks(conn)
            print(f"\nPopulation complete!")
            print(f"  Emails with tasks: {stats['emails_with_tasks']}")
            print(f"  Tasks inserted: {stats['tasks_inserted']}")
            print(f"  Tasks skipped: {stats['tasks_skipped']}")

        print_stats(conn)
    finally:
        conn.close()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
