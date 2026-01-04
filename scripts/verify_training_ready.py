#!/usr/bin/env python3
"""Gate 3: Verify training readiness before using data for ML training.

Runs comprehensive checks:
1. Data Quality Checks
   - Sufficient labeled data (>10k emails with action)
   - Balanced actions (not >80% single action type)
   - Response time distribution (reasonable spread)
   - Label diversity (multiple Gmail labels present)

2. Preference Pair Checks
   - Can generate >5k preference pairs
   - Pairs are meaningful (not trivial)
   - No self-pairs (same email in chosen/rejected)

3. Schema Integrity
   - All foreign keys valid
   - No orphan records
   - Indexes exist and used

4. Human Review Sample
   - Generate sample of 20 random emails for spot-check

Usage:
    python scripts/verify_training_ready.py
    python scripts/verify_training_ready.py --output results/gate3_result.json
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg


# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5433/rl_emails"

# Thresholds for gate checks
THRESHOLDS = {
    "min_emails_with_action": 10000,
    "max_single_action_pct": 80.0,
    "min_response_time_buckets": 3,
    "min_labels": 5,
    "min_preference_pairs": 5000,
    "human_review_sample_size": 20,
}

# Response time buckets (seconds)
RESPONSE_TIME_BUCKETS = {
    "IMMEDIATE": (0, 3600),           # < 1 hour
    "SAME_DAY": (3600, 86400),         # 1-24 hours
    "NEXT_DAY": (86400, 172800),       # 24-48 hours
    "LATER": (172800, float("inf")),   # > 48 hours
}

# Action ranking for preference pairs (higher = better)
ACTION_RANKING = {
    "REPLIED": 4,
    "FORWARDED": 3,
    "ARCHIVED": 2,
    "READ_PENDING": 1,
    "IGNORED": 0,
    "UNKNOWN": 0,
    "COMPOSED": None,  # Sent emails, not ranked
}

# Label priority for preference pairs (higher = better)
LABEL_PRIORITY = {
    "STARRED": 3,
    "IMPORTANT": 2,
    "CATEGORY_PERSONAL": 1,
    "INBOX": 0,
    "CATEGORY_UPDATES": 0,
    "CATEGORY_SOCIAL": 0,
    "CATEGORY_PROMOTIONS": -1,
    "CATEGORY_FORUMS": -1,
    "SPAM": -3,
    "TRASH": -3,
}


class GateChecker:
    """Runs Gate 3 training readiness checks."""

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self.conn: asyncpg.Connection = None
        self.checks: dict[str, dict] = {}
        self.issues: list[str] = []

    async def connect(self):
        """Connect to PostgreSQL."""
        self.conn = await asyncpg.connect(self.db_url)

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()

    # =========================================================================
    # Data Quality Checks
    # =========================================================================

    async def check_email_count_with_action(self) -> dict:
        """Check: Sufficient labeled data (>10k emails with action)."""
        result = await self.conn.fetchrow("""
            SELECT
                COUNT(*) as total_emails,
                COUNT(*) FILTER (WHERE action IS NOT NULL) as emails_with_action,
                COUNT(*) FILTER (WHERE action IS NOT NULL AND action != 'COMPOSED') as received_with_action
            FROM emails
        """)

        total = result["total_emails"]
        with_action = result["emails_with_action"]
        received = result["received_with_action"]
        threshold = THRESHOLDS["min_emails_with_action"]

        passed = received >= threshold

        check = {
            "name": "sufficient_labeled_data",
            "passed": passed,
            "threshold": threshold,
            "actual": received,
            "details": {
                "total_emails": total,
                "emails_with_action": with_action,
                "received_emails_with_action": received,
            },
        }

        if not passed:
            self.issues.append(
                f"Insufficient labeled data: {received} emails with action "
                f"(need {threshold})"
            )

        self.checks["sufficient_labeled_data"] = check
        return check

    async def check_action_balance(self) -> dict:
        """Check: Balanced actions (not >80% single action type)."""
        rows = await self.conn.fetch("""
            SELECT action, COUNT(*) as cnt
            FROM emails
            WHERE action IS NOT NULL AND action != 'COMPOSED'
            GROUP BY action
            ORDER BY cnt DESC
        """)

        total = sum(row["cnt"] for row in rows)
        distribution = {row["action"]: row["cnt"] for row in rows}

        # Check if any single action is > threshold
        max_pct = 0.0
        dominant_action = None
        percentages = {}

        for action, count in distribution.items():
            pct = 100.0 * count / total if total > 0 else 0
            percentages[action] = round(pct, 2)
            if pct > max_pct:
                max_pct = pct
                dominant_action = action

        threshold = THRESHOLDS["max_single_action_pct"]
        passed = max_pct <= threshold

        check = {
            "name": "action_balance",
            "passed": passed,
            "threshold": f"<={threshold}%",
            "actual": f"{max_pct:.1f}% ({dominant_action})",
            "details": {
                "distribution": distribution,
                "percentages": percentages,
                "dominant_action": dominant_action,
                "dominant_pct": round(max_pct, 2),
            },
        }

        if not passed:
            self.issues.append(
                f"Action imbalance: {dominant_action} is {max_pct:.1f}% "
                f"(max allowed {threshold}%)"
            )

        self.checks["action_balance"] = check
        return check

    async def check_response_time_distribution(self) -> dict:
        """Check: Response time distribution (reasonable spread)."""
        rows = await self.conn.fetch("""
            SELECT response_time_seconds
            FROM emails
            WHERE response_time_seconds IS NOT NULL
            AND response_time_seconds > 0
        """)

        # Bucket the response times
        buckets = {name: 0 for name in RESPONSE_TIME_BUCKETS}

        for row in rows:
            seconds = row["response_time_seconds"]
            for name, (low, high) in RESPONSE_TIME_BUCKETS.items():
                if low <= seconds < high:
                    buckets[name] += 1
                    break

        total = sum(buckets.values())
        non_empty_buckets = sum(1 for count in buckets.values() if count > 0)
        threshold = THRESHOLDS["min_response_time_buckets"]

        passed = non_empty_buckets >= threshold

        percentages = {}
        for name, count in buckets.items():
            pct = 100.0 * count / total if total > 0 else 0
            percentages[name] = round(pct, 2)

        check = {
            "name": "response_time_distribution",
            "passed": passed,
            "threshold": f">={threshold} buckets with data",
            "actual": f"{non_empty_buckets} buckets",
            "details": {
                "total_with_response_time": total,
                "buckets": buckets,
                "percentages": percentages,
                "non_empty_buckets": non_empty_buckets,
            },
        }

        if not passed:
            self.issues.append(
                f"Response time distribution too narrow: {non_empty_buckets} buckets "
                f"(need {threshold})"
            )

        self.checks["response_time_distribution"] = check
        return check

    async def check_label_diversity(self) -> dict:
        """Check: Label diversity (multiple Gmail labels present)."""
        rows = await self.conn.fetch("""
            SELECT unnest(labels) as label, COUNT(*) as cnt
            FROM emails
            WHERE labels IS NOT NULL
            GROUP BY label
            ORDER BY cnt DESC
        """)

        labels = {row["label"]: row["cnt"] for row in rows}
        unique_labels = len(labels)
        threshold = THRESHOLDS["min_labels"]

        passed = unique_labels >= threshold

        check = {
            "name": "label_diversity",
            "passed": passed,
            "threshold": f">={threshold} unique labels",
            "actual": f"{unique_labels} labels",
            "details": {
                "unique_labels": unique_labels,
                "top_labels": dict(list(labels.items())[:15]),
            },
        }

        if not passed:
            self.issues.append(
                f"Insufficient label diversity: {unique_labels} labels (need {threshold})"
            )

        self.checks["label_diversity"] = check
        return check

    # =========================================================================
    # Preference Pair Checks
    # =========================================================================

    async def check_preference_pairs(self) -> dict:
        """Check: Can generate >5k preference pairs."""
        # Estimate preference pairs from action ranking
        action_rows = await self.conn.fetch("""
            SELECT action, COUNT(*) as cnt
            FROM emails
            WHERE action IS NOT NULL AND action != 'COMPOSED'
            GROUP BY action
        """)

        action_counts = {row["action"]: row["cnt"] for row in action_rows}

        # Count potential action-based pairs
        action_pairs = 0
        actions_ranked = [a for a in action_counts if ACTION_RANKING.get(a) is not None]

        for i, action1 in enumerate(actions_ranked):
            for action2 in actions_ranked[i + 1:]:
                rank1 = ACTION_RANKING.get(action1, 0)
                rank2 = ACTION_RANKING.get(action2, 0)
                if rank1 != rank2:
                    count1 = action_counts.get(action1, 0)
                    count2 = action_counts.get(action2, 0)
                    # Estimated pairs (sample without replacement)
                    action_pairs += min(count1, count2, 1000)

        # Estimate label-based pairs
        label_rows = await self.conn.fetch("""
            SELECT unnest(labels) as label, COUNT(*) as cnt
            FROM emails
            WHERE labels IS NOT NULL AND action IS NOT NULL AND action != 'COMPOSED'
            GROUP BY label
        """)

        label_counts = {row["label"]: row["cnt"] for row in label_rows}

        label_pairs = 0
        for label1, count1 in label_counts.items():
            prio1 = LABEL_PRIORITY.get(label1.upper(), 0)
            for label2, count2 in label_counts.items():
                prio2 = LABEL_PRIORITY.get(label2.upper(), 0)
                if prio1 > prio2:
                    label_pairs += min(count1, count2, 500)

        # Estimate response-time pairs
        rt_rows = await self.conn.fetch("""
            SELECT
                COUNT(*) FILTER (WHERE response_time_seconds < 3600) as quick,
                COUNT(*) FILTER (WHERE response_time_seconds >= 3600) as slow
            FROM emails
            WHERE response_time_seconds IS NOT NULL AND response_time_seconds > 0
        """)

        quick = rt_rows[0]["quick"]
        slow = rt_rows[0]["slow"]
        response_time_pairs = min(quick, slow, 2000)

        total_estimated = action_pairs + label_pairs + response_time_pairs
        threshold = THRESHOLDS["min_preference_pairs"]

        passed = total_estimated >= threshold

        check = {
            "name": "preference_pair_count",
            "passed": passed,
            "threshold": f">={threshold} pairs",
            "actual": f"{total_estimated} estimated pairs",
            "details": {
                "estimated_action_pairs": action_pairs,
                "estimated_label_pairs": label_pairs,
                "estimated_response_time_pairs": response_time_pairs,
                "total_estimated": total_estimated,
            },
        }

        if not passed:
            self.issues.append(
                f"Insufficient preference pairs: {total_estimated} estimated (need {threshold})"
            )

        self.checks["preference_pair_count"] = check
        return check

    async def check_no_self_pairs(self) -> dict:
        """Check: No self-pairs in potential pair generation.

        This is a sanity check - the pair generation logic should never
        produce self-pairs, but we verify the data supports this.
        """
        # Check for duplicate message_ids (which could cause self-pairs)
        dup_rows = await self.conn.fetch("""
            SELECT message_id, COUNT(*) as cnt
            FROM emails
            WHERE action IS NOT NULL
            GROUP BY message_id
            HAVING COUNT(*) > 1
        """)

        duplicate_count = len(dup_rows)
        passed = duplicate_count == 0

        check = {
            "name": "no_self_pairs",
            "passed": passed,
            "threshold": "0 duplicate message_ids",
            "actual": f"{duplicate_count} duplicates",
            "details": {
                "duplicate_message_ids": [row["message_id"] for row in dup_rows[:10]],
            },
        }

        if not passed:
            self.issues.append(
                f"Found {duplicate_count} duplicate message_ids - could cause self-pairs"
            )

        self.checks["no_self_pairs"] = check
        return check

    # =========================================================================
    # Schema Integrity Checks
    # =========================================================================

    async def check_foreign_keys(self) -> dict:
        """Check: All foreign keys are valid."""
        # Check emails.raw_email_id references valid raw_emails.id
        orphan_emails = await self.conn.fetchval("""
            SELECT COUNT(*)
            FROM emails e
            LEFT JOIN raw_emails r ON e.raw_email_id = r.id
            WHERE e.raw_email_id IS NOT NULL AND r.id IS NULL
        """)

        # Check attachments.raw_email_id
        orphan_attachments_raw = await self.conn.fetchval("""
            SELECT COUNT(*)
            FROM attachments a
            LEFT JOIN raw_emails r ON a.raw_email_id = r.id
            WHERE a.raw_email_id IS NOT NULL AND r.id IS NULL
        """)

        # Check attachments.email_id
        orphan_attachments_email = await self.conn.fetchval("""
            SELECT COUNT(*)
            FROM attachments a
            LEFT JOIN emails e ON a.email_id = e.id
            WHERE a.email_id IS NOT NULL AND e.id IS NULL
        """)

        total_orphans = orphan_emails + orphan_attachments_raw + orphan_attachments_email
        passed = total_orphans == 0

        check = {
            "name": "foreign_keys_valid",
            "passed": passed,
            "threshold": "0 orphan records",
            "actual": f"{total_orphans} orphans",
            "details": {
                "orphan_emails": orphan_emails,
                "orphan_attachments_raw": orphan_attachments_raw,
                "orphan_attachments_email": orphan_attachments_email,
            },
        }

        if not passed:
            self.issues.append(f"Found {total_orphans} orphan records with invalid foreign keys")

        self.checks["foreign_keys_valid"] = check
        return check

    async def check_no_orphan_records(self) -> dict:
        """Check: No orphan records in aggregation tables."""
        # Check if threads reference valid thread_ids
        orphan_threads = await self.conn.fetchval("""
            SELECT COUNT(*)
            FROM threads t
            WHERE NOT EXISTS (
                SELECT 1 FROM emails e WHERE e.thread_id = t.thread_id
            )
        """)

        # Check if users reference valid emails
        orphan_users = await self.conn.fetchval("""
            SELECT COUNT(*)
            FROM users u
            WHERE u.emails_from = 0 AND u.emails_to = 0 AND NOT u.is_you
        """)

        total_orphans = orphan_threads + orphan_users
        passed = total_orphans == 0

        check = {
            "name": "no_orphan_records",
            "passed": passed,
            "threshold": "0 orphan aggregation records",
            "actual": f"{total_orphans} orphans",
            "details": {
                "orphan_threads": orphan_threads,
                "orphan_users": orphan_users,
            },
        }

        if not passed:
            self.issues.append(f"Found {total_orphans} orphan records in aggregation tables")

        self.checks["no_orphan_records"] = check
        return check

    async def check_indexes_exist(self) -> dict:
        """Check: Required indexes exist."""
        # Query pg_indexes for our tables
        rows = await self.conn.fetch("""
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('emails', 'raw_emails', 'attachments', 'threads', 'users')
            ORDER BY tablename, indexname
        """)

        indexes_by_table = {}
        for row in rows:
            table = row["tablename"]
            if table not in indexes_by_table:
                indexes_by_table[table] = []
            indexes_by_table[table].append(row["indexname"])

        # Required indexes for training queries
        required_indexes = {
            "emails": [
                "idx_emails_thread_id",
                "idx_emails_date",
                "idx_emails_labels",
            ],
            "raw_emails": [
                "idx_raw_emails_message_id",
            ],
        }

        missing = []
        for table, required in required_indexes.items():
            existing = indexes_by_table.get(table, [])
            for idx in required:
                if idx not in existing:
                    missing.append(f"{table}.{idx}")

        passed = len(missing) == 0

        check = {
            "name": "indexes_exist",
            "passed": passed,
            "threshold": "All required indexes present",
            "actual": f"{len(missing)} missing" if missing else "All present",
            "details": {
                "indexes_by_table": indexes_by_table,
                "missing_indexes": missing,
            },
        }

        if not passed:
            self.issues.append(f"Missing indexes: {', '.join(missing)}")

        self.checks["indexes_exist"] = check
        return check

    # =========================================================================
    # Human Review Sample
    # =========================================================================

    async def generate_human_review_sample(self) -> list[dict]:
        """Generate sample of 20 random emails for human spot-check."""
        sample_size = THRESHOLDS["human_review_sample_size"]

        rows = await self.conn.fetch(f"""
            SELECT
                e.id,
                e.message_id,
                e.thread_id,
                e.subject,
                e.from_email,
                e.from_name,
                e.to_emails,
                e.date_parsed,
                e.body_preview,
                e.labels,
                e.action,
                e.timing,
                e.response_time_seconds,
                e.has_attachments,
                e.attachment_count,
                e.is_sent
            FROM emails e
            WHERE e.action IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """)

        sample = []
        for row in rows:
            sample.append({
                "id": row["id"],
                "message_id": row["message_id"],
                "thread_id": row["thread_id"],
                "subject": row["subject"],
                "from_email": row["from_email"],
                "from_name": row["from_name"],
                "to_emails": row["to_emails"],
                "date_parsed": row["date_parsed"].isoformat() if row["date_parsed"] else None,
                "body_preview": row["body_preview"][:200] if row["body_preview"] else None,
                "labels": row["labels"],
                "action": row["action"],
                "timing": row["timing"],
                "response_time_seconds": row["response_time_seconds"],
                "has_attachments": row["has_attachments"],
                "attachment_count": row["attachment_count"],
                "is_sent": row["is_sent"],
            })

        return sample

    # =========================================================================
    # Main Gate Check
    # =========================================================================

    async def run_all_checks(self) -> dict[str, Any]:
        """Run all gate checks and return result."""
        print("Gate 3: Training Readiness Check")
        print("=" * 50)

        # Data Quality Checks
        print("\n[1/4] Data Quality Checks")
        print("-" * 30)

        print("  Checking email count with action...")
        await self.check_email_count_with_action()
        print(f"    -> {'PASS' if self.checks['sufficient_labeled_data']['passed'] else 'FAIL'}")

        print("  Checking action balance...")
        await self.check_action_balance()
        print(f"    -> {'PASS' if self.checks['action_balance']['passed'] else 'FAIL'}")

        print("  Checking response time distribution...")
        await self.check_response_time_distribution()
        print(f"    -> {'PASS' if self.checks['response_time_distribution']['passed'] else 'FAIL'}")

        print("  Checking label diversity...")
        await self.check_label_diversity()
        print(f"    -> {'PASS' if self.checks['label_diversity']['passed'] else 'FAIL'}")

        # Preference Pair Checks
        print("\n[2/4] Preference Pair Checks")
        print("-" * 30)

        print("  Estimating preference pair count...")
        await self.check_preference_pairs()
        print(f"    -> {'PASS' if self.checks['preference_pair_count']['passed'] else 'FAIL'}")

        print("  Checking for self-pair risk...")
        await self.check_no_self_pairs()
        print(f"    -> {'PASS' if self.checks['no_self_pairs']['passed'] else 'FAIL'}")

        # Schema Integrity Checks
        print("\n[3/4] Schema Integrity Checks")
        print("-" * 30)

        print("  Checking foreign key validity...")
        await self.check_foreign_keys()
        print(f"    -> {'PASS' if self.checks['foreign_keys_valid']['passed'] else 'FAIL'}")

        print("  Checking for orphan records...")
        await self.check_no_orphan_records()
        print(f"    -> {'PASS' if self.checks['no_orphan_records']['passed'] else 'FAIL'}")

        print("  Checking index existence...")
        await self.check_indexes_exist()
        print(f"    -> {'PASS' if self.checks['indexes_exist']['passed'] else 'FAIL'}")

        # Human Review Sample
        print("\n[4/4] Human Review Sample")
        print("-" * 30)

        print(f"  Generating {THRESHOLDS['human_review_sample_size']} sample emails...")
        human_review_sample = await self.generate_human_review_sample()
        print(f"    -> Generated {len(human_review_sample)} samples")

        # Determine overall status
        all_passed = all(check["passed"] for check in self.checks.values())

        if all_passed:
            status = "PASS"
            recommendation = "Ready for training"
        elif len(self.issues) <= 2:
            status = "NEEDS_REVIEW"
            recommendation = "Minor issues found - manual review recommended"
        else:
            status = "FAIL"
            recommendation = "Multiple issues found - not ready for training"

        # Build result
        result = {
            "gate": "training_ready",
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "checks": self.checks,
            "issues": self.issues,
            "human_review_sample": human_review_sample,
            "recommendation": recommendation,
            "thresholds": THRESHOLDS,
        }

        return result


async def main():
    parser = argparse.ArgumentParser(
        description="Gate 3: Verify training readiness"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file for gate result",
    )
    parser.add_argument(
        "--db-url",
        default=DB_URL,
        help=f"PostgreSQL connection URL (default: {DB_URL})",
    )

    args = parser.parse_args()

    checker = GateChecker(db_url=args.db_url)

    try:
        print(f"Connecting to {args.db_url}...")
        await checker.connect()

        result = await checker.run_all_checks()

        # Print summary
        print("\n" + "=" * 50)
        print("GATE 3 RESULT")
        print("=" * 50)
        print(f"Status: {result['status']}")
        print(f"Recommendation: {result['recommendation']}")

        if result["issues"]:
            print(f"\nIssues ({len(result['issues'])}):")
            for issue in result["issues"]:
                print(f"  - {issue}")

        # Write output
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResult written to: {args.output}")
        else:
            # Print JSON to stdout
            print("\n" + "-" * 50)
            print("JSON Result:")
            print(json.dumps(result, indent=2, default=str))

        # Exit with appropriate code
        if result["status"] == "PASS":
            sys.exit(0)
        elif result["status"] == "NEEDS_REVIEW":
            sys.exit(1)
        else:
            sys.exit(2)

    except asyncpg.exceptions.ConnectionDoesNotExistError:
        print(f"Error: Could not connect to database at {args.db_url}")
        print("Make sure PostgreSQL is running and the database exists.")
        sys.exit(3)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(3)
    finally:
        await checker.close()


if __name__ == "__main__":
    asyncio.run(main())
