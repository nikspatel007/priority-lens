"""Entity extraction service for projects, tasks, and priority contexts.

Extracts structured entities from existing pipeline data:
- Tasks from LLM classifications and AI classifications (REAL PEOPLE ONLY)
- Projects from cluster metadata (requires real person majority)
- Priority contexts from email features and relationships

Key principle: Focus on emails from real people, filter out marketing/service emails.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

# Patterns that indicate marketing/automated/service emails
MARKETING_PATTERNS = [
    "%noreply%",
    "%no-reply%",
    "%no_reply%",
    "%donotreply%",
    "%@email.%",
    "%@mail.%",
    "%@offers.%",
    "%@mailer.%",
    "%@marketing.%",
    "%@news.%",
    "%@info.%",
    "%newsletter%",
    "%marketing%",
    "%alerts@%",
    "%notification%",
    "%billing@%",
    "%digest%",
    "%@google.com",
    "%@github.com",
    "%@linkedin.com",
    "%@medium.com",
    "%@substack.%",
    "%@mailchimp.%",
    "%@sendgrid.%",
    "%update@%",
    "%support@%",
    "%team@%",
    "%hello@%",
    "%contact@%",
]


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    tasks_created: int
    tasks_updated: int
    projects_created: int
    projects_updated: int
    priority_contexts_created: int
    priority_contexts_updated: int
    errors: list[str]
    # New: tracking filtered items
    tasks_filtered_marketing: int = 0
    projects_filtered_marketing: int = 0

    @property
    def total_created(self) -> int:
        """Total entities created."""
        return self.tasks_created + self.projects_created + self.priority_contexts_created

    @property
    def total_updated(self) -> int:
        """Total entities updated."""
        return self.tasks_updated + self.projects_updated + self.priority_contexts_updated


def _build_marketing_filter_sql() -> str:
    """Build SQL WHERE clause to filter out marketing emails."""
    conditions = []
    for pattern in MARKETING_PATTERNS:
        conditions.append(f"e.from_email NOT LIKE '{pattern}'")
    return " AND ".join(conditions)


def _is_real_person_sql() -> str:
    """SQL CASE expression to determine if sender is a real person."""
    marketing_checks = " OR ".join([f"e.from_email LIKE '{p}'" for p in MARKETING_PATTERNS])
    return f"""
        CASE
            WHEN {marketing_checks} THEN false
            WHEN u.reply_rate IS NOT NULL AND u.reply_rate > 0 THEN true
            WHEN u.is_important_sender = true THEN true
            WHEN u.emails_from IS NOT NULL AND u.emails_from <= 3
                 AND u.emails_from > 0 THEN true
            ELSE false
        END
    """


class TaskExtractor:
    """Extract tasks from LLM and AI classifications (real people only)."""

    def __init__(self, conn: Connection, user_id: uuid.UUID | None = None) -> None:
        """Initialize task extractor.

        Args:
            conn: Database connection.
            user_id: Optional user ID for multi-tenant mode.
        """
        self.conn = conn
        self.user_id = user_id

    def extract_from_llm_classifications(self) -> tuple[int, int, int, list[str]]:
        """Extract tasks from LLM classification results (real people only).

        Returns:
            Tuple of (created_count, updated_count, filtered_count, errors).
        """
        created = 0
        updated = 0
        filtered = 0
        errors: list[str] = []

        is_real_person = _is_real_person_sql()

        # Get actionable emails from LLM classification - ONLY from real people
        query = text(
            f"""
            SELECT
                llc.email_id,
                llc.action_type,
                llc.urgency,
                llc.next_step,
                llc.one_liner,
                e.subject,
                e.from_email,
                u.name as sender_name,
                u.reply_rate,
                ({is_real_person}) as is_real_person
            FROM email_llm_classification llc
            JOIN emails e ON e.id = llc.email_id
            LEFT JOIN users u ON u.email = e.from_email
            WHERE llc.action_type IN ('task', 'reply', 'decision')
        """
        )

        rows = self.conn.execute(query).fetchall()

        for row in rows:
            email_id = row[0]
            action_type = row[1]
            urgency = row[2]
            _next_step = row[3]
            one_liner = row[4]
            subject = row[5]
            from_email = row[6]
            sender_name = row[7]
            reply_rate = row[8]
            is_real_person = row[9]

            # FILTER: Skip if not from a real person
            if not is_real_person:
                filtered += 1
                continue

            # Map urgency to urgency_score
            urgency_map = {
                "immediate": 1.0,
                "today": 0.8,
                "this_week": 0.5,
                "whenever": 0.2,
                "none": 0.1,
            }
            urgency_score = urgency_map.get(urgency or "none", 0.1)

            # Boost urgency for people you've replied to frequently
            if reply_rate and reply_rate > 0.5:
                urgency_score = min(1.0, urgency_score + 0.1)

            # Map action_type to task_type
            task_type_map = {
                "task": "other",
                "reply": "send",
                "decision": "decision",
            }
            task_type = task_type_map.get(action_type or "other", "other")

            # Generate unique task_id
            task_id = f"llm_{email_id}_{action_type}"

            # Check if task exists
            existing = self.conn.execute(
                text("SELECT id FROM tasks WHERE task_id = :task_id"),
                {"task_id": task_id},
            ).fetchone()

            # Build description with sender context
            description = one_liner or subject or "Task from email"
            if sender_name and sender_name != from_email:
                description = f"[{sender_name}] {description}"
            if len(description) > 1000:
                description = description[:997] + "..."

            if existing:
                try:
                    self.conn.execute(
                        text(
                            """
                            UPDATE tasks SET
                                description = :description,
                                urgency_score = :urgency_score,
                                task_type = :task_type,
                                extraction_method = :extraction_method,
                                assigned_by = :assigned_by,
                                status = COALESCE(status, 'pending')
                            WHERE task_id = :task_id
                        """
                        ),
                        {
                            "task_id": task_id,
                            "description": description,
                            "urgency_score": urgency_score,
                            "task_type": task_type,
                            "extraction_method": "llm_real_person",
                            "assigned_by": from_email,
                        },
                    )
                    updated += 1
                except Exception as e:
                    errors.append(f"Error updating task {task_id}: {e}")
            else:
                try:
                    self.conn.execute(
                        text(
                            """
                            INSERT INTO tasks (
                                task_id, email_id, description, task_type,
                                urgency_score, extraction_method, status,
                                assigned_by, user_id
                            ) VALUES (
                                :task_id, :email_id, :description, :task_type,
                                :urgency_score, :extraction_method, :status,
                                :assigned_by, :user_id
                            )
                        """
                        ),
                        {
                            "task_id": task_id,
                            "email_id": email_id,
                            "description": description,
                            "task_type": task_type,
                            "urgency_score": urgency_score,
                            "extraction_method": "llm_real_person",
                            "status": "pending",
                            "assigned_by": from_email,
                            "user_id": str(self.user_id) if self.user_id else None,
                        },
                    )
                    created += 1
                except Exception as e:
                    errors.append(f"Error creating task {task_id}: {e}")

        return created, updated, filtered, errors

    def extract_from_ai_classifications(self) -> tuple[int, int, int, list[str]]:
        """Extract tasks from AI classification signals (real people only).

        Returns:
            Tuple of (created_count, updated_count, filtered_count, errors).
        """
        created = 0
        updated = 0
        filtered = 0
        errors: list[str] = []

        is_real_person = _is_real_person_sql()

        # Get emails with actionable signals - ONLY from real people
        query = text(
            f"""
            SELECT
                eac.email_id,
                eac.has_question,
                eac.has_request,
                eac.has_deadline,
                eac.has_approval,
                eac.has_scheduling,
                e.subject,
                e.from_email,
                u.name as sender_name,
                u.reply_rate,
                ({is_real_person}) as is_real_person
            FROM email_ai_classification eac
            JOIN emails e ON e.id = eac.email_id
            LEFT JOIN users u ON u.email = e.from_email
            WHERE (eac.has_request = true OR eac.has_deadline = true
                   OR eac.has_approval = true)
            AND NOT EXISTS (
                SELECT 1 FROM tasks t WHERE t.email_id = eac.email_id
            )
        """
        )

        rows = self.conn.execute(query).fetchall()

        for row in rows:
            email_id = row[0]
            _has_question = row[1]
            has_request = row[2]
            has_deadline = row[3]
            has_approval = row[4]
            _has_scheduling = row[5]
            subject = row[6]
            from_email = row[7]
            sender_name = row[8]
            reply_rate = row[9]
            is_real_person = row[10]

            # FILTER: Skip if not from a real person
            if not is_real_person:
                filtered += 1
                continue

            # Determine task type from signals
            if has_approval:
                task_type = "decision"
            elif has_request:
                task_type = "follow_up"
            elif has_deadline:
                task_type = "other"
            else:
                task_type = "other"

            # Calculate urgency based on signals and relationship
            urgency_score = 0.4  # Base score for real person
            if has_deadline:
                urgency_score = 0.7
            if has_approval:
                urgency_score = max(urgency_score, 0.6)

            # Boost for high reply rate contacts
            if reply_rate and reply_rate > 0.5:
                urgency_score = min(1.0, urgency_score + 0.1)

            task_id = f"ai_{email_id}"

            # Build description with sender context
            description = subject or "Task detected from email"
            if sender_name and sender_name != from_email:
                description = f"[{sender_name}] {description}"
            if len(description) > 1000:
                description = description[:997] + "..."

            try:
                self.conn.execute(
                    text(
                        """
                        INSERT INTO tasks (
                            task_id, email_id, description, task_type,
                            urgency_score, extraction_method, status,
                            assigned_by, user_id
                        ) VALUES (
                            :task_id, :email_id, :description, :task_type,
                            :urgency_score, :extraction_method, :status,
                            :assigned_by, :user_id
                        )
                    """
                    ),
                    {
                        "task_id": task_id,
                        "email_id": email_id,
                        "description": description,
                        "task_type": task_type,
                        "urgency_score": urgency_score,
                        "extraction_method": "ai_real_person",
                        "status": "pending",
                        "assigned_by": from_email,
                        "user_id": str(self.user_id) if self.user_id else None,
                    },
                )
                created += 1
            except Exception as e:
                errors.append(f"Error creating AI task for email {email_id}: {e}")

        return created, updated, filtered, errors


class ProjectExtractor:
    """Extract projects from cluster metadata (real person majority required)."""

    def __init__(self, conn: Connection, user_id: uuid.UUID | None = None) -> None:
        """Initialize project extractor.

        Args:
            conn: Database connection.
            user_id: Optional user ID for multi-tenant mode.
        """
        self.conn = conn
        self.user_id = user_id

    def extract_from_real_person_threads(self) -> tuple[int, int, int, list[str]]:
        """Extract projects from real person conversation threads.

        This is the primary method - finds actual conversations with people
        you've communicated with.

        Returns:
            Tuple of (created_count, updated_count, filtered_count, errors).
        """
        created = 0
        updated = 0
        filtered = 0
        errors: list[str] = []

        # Find conversation threads with real people (you've replied to them)
        query = text(
            """
            SELECT
                u.email as contact_email,
                u.name as contact_name,
                u.reply_rate,
                u.is_important_sender,
                count(DISTINCT e.id) as email_count,
                count(DISTINCT e.thread_id) as thread_count,
                max(e.date_parsed) as last_activity,
                array_agg(DISTINCT left(e.subject, 100)) as subjects
            FROM users u
            JOIN emails e ON e.from_email = u.email
            WHERE u.reply_rate > 0
              AND u.is_you = false
              AND e.is_sent = false
            GROUP BY u.email, u.name, u.reply_rate, u.is_important_sender
            HAVING count(DISTINCT e.id) >= 2
            ORDER BY count(DISTINCT e.id) DESC
        """
        )

        rows = self.conn.execute(query).fetchall()

        for row in rows:
            contact_email = row[0]
            contact_name = row[1]
            reply_rate = row[2]
            is_important = row[3]
            email_count = row[4]
            _thread_count = row[5]
            last_activity = row[6]
            subjects = row[7]

            # Generate project name from contact
            if contact_name and contact_name != contact_email:
                project_name = f"Conversation: {contact_name}"
            else:
                # Use email prefix
                email_prefix = contact_email.split("@")[0]
                project_name = f"Conversation: {email_prefix}"

            # Try to extract topic from subjects
            if subjects and len(subjects) > 0:
                # Find common words in subjects (simple approach)
                first_subject = subjects[0] if subjects[0] else ""
                if first_subject and len(first_subject) > 5:
                    # Use first meaningful subject as topic hint
                    topic = first_subject[:50]
                    if "Re:" not in topic and "Fwd:" not in topic:
                        project_name = f"{project_name} - {topic}"

            if len(project_name) > 200:
                project_name = project_name[:197] + "..."

            # Determine project type
            if is_important:
                project_type = "key_relationship"
            elif reply_rate and reply_rate > 0.5:
                project_type = "active_conversation"
            else:
                project_type = "conversation"

            # Calculate confidence based on engagement
            confidence = 0.7
            if reply_rate and reply_rate > 0.5:
                confidence = 0.9
            if is_important:
                confidence = 0.95

            # Check if project exists for this contact
            existing = self.conn.execute(
                text(
                    """
                    SELECT id FROM projects
                    WHERE owner_email = :email
                    AND detected_from = 'real_person_thread'
                """
                ),
                {"email": contact_email},
            ).fetchone()

            if existing:
                try:
                    self.conn.execute(
                        text(
                            """
                            UPDATE projects SET
                                name = :name,
                                email_count = :email_count,
                                last_activity = :last_activity,
                                confidence = :confidence,
                                project_type = :project_type
                            WHERE owner_email = :email
                            AND detected_from = 'real_person_thread'
                        """
                        ),
                        {
                            "name": project_name,
                            "email": contact_email,
                            "email_count": email_count,
                            "last_activity": last_activity,
                            "confidence": confidence,
                            "project_type": project_type,
                        },
                    )
                    updated += 1
                except Exception as e:
                    errors.append(f"Error updating project for {contact_email}: {e}")
            else:
                try:
                    self.conn.execute(
                        text(
                            """
                            INSERT INTO projects (
                                name, project_type, owner_email, email_count,
                                last_activity, detected_from, confidence, user_id
                            ) VALUES (
                                :name, :project_type, :owner_email, :email_count,
                                :last_activity, :detected_from, :confidence, :user_id
                            )
                        """
                        ),
                        {
                            "name": project_name,
                            "project_type": project_type,
                            "owner_email": contact_email,
                            "email_count": email_count,
                            "last_activity": last_activity,
                            "detected_from": "real_person_thread",
                            "confidence": confidence,
                            "user_id": str(self.user_id) if self.user_id else None,
                        },
                    )
                    created += 1
                except Exception as e:
                    errors.append(f"Error creating project for {contact_email}: {e}")

        return created, updated, filtered, errors

    def extract_from_content_clusters(self) -> tuple[int, int, int, list[str]]:
        """Extract projects from content clusters (real person majority required).

        Only creates projects from clusters where >50% of emails are from real people.

        Returns:
            Tuple of (created_count, updated_count, filtered_count, errors).
        """
        created = 0
        updated = 0
        filtered = 0
        errors: list[str] = []

        is_real_person = _is_real_person_sql()

        # Get content clusters with real person ratio
        query = text(
            f"""
            WITH cluster_analysis AS (
                SELECT
                    cm.cluster_id,
                    cm.size,
                    cm.auto_label,
                    cm.last_activity_at,
                    count(DISTINCT e.id) as email_count,
                    sum(CASE WHEN ({is_real_person}) THEN 1 ELSE 0 END) as real_person_count,
                    array_agg(DISTINCT e.from_email) as senders,
                    array_agg(DISTINCT left(e.subject, 60)) as subjects
                FROM cluster_metadata cm
                JOIN email_clusters ec ON ec.content_cluster_id = cm.cluster_id
                JOIN emails e ON e.id = ec.email_id
                LEFT JOIN users u ON u.email = e.from_email
                WHERE cm.dimension = 'content'
                AND cm.size >= 3
                GROUP BY cm.cluster_id, cm.size, cm.auto_label, cm.last_activity_at
            )
            SELECT
                cluster_id,
                size,
                auto_label,
                last_activity_at,
                email_count,
                real_person_count,
                senders,
                subjects,
                (real_person_count::float / NULLIF(email_count, 0)) as real_person_ratio
            FROM cluster_analysis
            WHERE real_person_count > 0
            ORDER BY real_person_ratio DESC, size DESC
        """
        )

        rows = self.conn.execute(query).fetchall()

        for row in rows:
            cluster_id = row[0]
            _size = row[1]
            auto_label = row[2]
            last_activity = row[3]
            email_count = row[4]
            _real_person_count = row[5]
            senders = row[6]
            subjects = row[7]
            real_person_ratio = row[8]

            # FILTER: Require majority real person emails
            if real_person_ratio < 0.5:
                filtered += 1
                continue

            # Generate project name
            if auto_label:
                project_name = auto_label
            elif subjects and subjects[0]:
                project_name = subjects[0]
            else:
                project_name = f"Topic Cluster {cluster_id}"

            if len(project_name) > 200:
                project_name = project_name[:197] + "..."

            # Check if we already have this as a real_person_thread project
            # (avoid duplicates)
            if senders:
                sender_check = self.conn.execute(
                    text(
                        """
                        SELECT id FROM projects
                        WHERE owner_email = ANY(:senders)
                        AND detected_from = 'real_person_thread'
                    """
                    ),
                    {"senders": list(senders)},
                ).fetchone()
                if sender_check:
                    # Already covered by real person thread detection
                    filtered += 1
                    continue

            project_type = "topic_cluster"
            confidence = min(0.9, 0.5 + real_person_ratio * 0.4)

            existing = self.conn.execute(
                text("SELECT id FROM projects WHERE cluster_id = :cluster_id"),
                {"cluster_id": cluster_id},
            ).fetchone()

            if existing:
                try:
                    self.conn.execute(
                        text(
                            """
                            UPDATE projects SET
                                name = :name,
                                email_count = :email_count,
                                last_activity = :last_activity,
                                confidence = :confidence
                            WHERE cluster_id = :cluster_id
                        """
                        ),
                        {
                            "name": project_name,
                            "cluster_id": cluster_id,
                            "email_count": email_count,
                            "last_activity": last_activity,
                            "confidence": confidence,
                        },
                    )
                    updated += 1
                except Exception as e:
                    errors.append(f"Error updating cluster project {cluster_id}: {e}")
            else:
                try:
                    self.conn.execute(
                        text(
                            """
                            INSERT INTO projects (
                                name, project_type, email_count, last_activity,
                                cluster_id, detected_from, confidence, user_id
                            ) VALUES (
                                :name, :project_type, :email_count, :last_activity,
                                :cluster_id, :detected_from, :confidence, :user_id
                            )
                        """
                        ),
                        {
                            "name": project_name,
                            "project_type": project_type,
                            "email_count": email_count,
                            "last_activity": last_activity,
                            "cluster_id": cluster_id,
                            "detected_from": "content_cluster_real",
                            "confidence": confidence,
                            "user_id": str(self.user_id) if self.user_id else None,
                        },
                    )
                    created += 1
                except Exception as e:
                    errors.append(f"Error creating cluster project {cluster_id}: {e}")

        return created, updated, filtered, errors


class PriorityContextBuilder:
    """Build priority contexts with real person weighting."""

    def __init__(self, conn: Connection, user_id: uuid.UUID | None = None) -> None:
        """Initialize priority context builder.

        Args:
            conn: Database connection.
            user_id: Optional user ID for multi-tenant mode.
        """
        self.conn = conn
        self.user_id = user_id

    def build_contexts(self) -> tuple[int, int, list[str]]:
        """Build priority contexts for all emails with real person weighting.

        Returns:
            Tuple of (created_count, updated_count, errors).
        """
        created = 0
        updated = 0
        errors: list[str] = []

        is_real_person = _is_real_person_sql()

        # Get emails with priority data and real person flag
        query = text(
            f"""
            SELECT
                ep.email_id,
                ep.feature_score,
                ep.replied_similarity,
                ep.cluster_novelty,
                ep.sender_novelty,
                ep.priority_score,
                e.from_email,
                e.thread_id,
                e.date_parsed,
                u.reply_rate,
                u.emails_from,
                u.is_important_sender,
                t.email_count as thread_length,
                ({is_real_person}) as is_real_person
            FROM email_priority ep
            JOIN emails e ON e.id = ep.email_id
            LEFT JOIN users u ON u.email = e.from_email
            LEFT JOIN threads t ON t.thread_id = e.thread_id
        """
        )

        rows = self.conn.execute(query).fetchall()

        now = datetime.now(UTC)

        for row in rows:
            email_id = row[0]
            _feature_score = row[1]
            _replied_similarity = row[2]
            _cluster_novelty = row[3]
            sender_novelty = row[4]
            priority_score = row[5]
            from_email = row[6]
            thread_id = row[7]
            date_parsed = row[8]
            reply_rate = row[9]
            emails_from = row[10]
            is_important = row[11]
            thread_length = row[12]
            is_real_person = row[13]

            # Calculate age in hours
            age_hours = None
            if date_parsed:
                delta = now - date_parsed.replace(tzinfo=UTC)
                age_hours = delta.total_seconds() / 3600

            # Determine business hours
            is_business_hours = None
            if date_parsed:
                hour = date_parsed.hour
                weekday = date_parsed.weekday()
                is_business_hours = 9 <= hour <= 17 and weekday < 5

            # Calculate component scores with REAL PERSON WEIGHTING

            # People score - heavily weight real people
            people_score = 0.0
            if is_real_person:
                people_score = 0.5  # Base score for being a real person
                if is_important:
                    people_score += 0.3
                if reply_rate and reply_rate > 0.5:
                    people_score += 0.2
            else:
                # Marketing/service emails get low people score
                people_score = 0.1

            # Temporal score
            temporal_score = 0.0
            if age_hours is not None:
                if age_hours < 24:
                    temporal_score = 0.8
                elif age_hours < 72:
                    temporal_score = 0.5
                else:
                    temporal_score = 0.2

            # Relationship score - based on actual relationship
            relationship_score = 0.0
            if is_real_person:
                relationship_score = 0.5
                if reply_rate and reply_rate > 0:
                    relationship_score += min(0.5, reply_rate)
            else:
                relationship_score = 1.0 - (sender_novelty or 0.5)

            # Calculate adjusted overall priority
            # Real people get a significant boost
            if is_real_person:
                adjusted_priority = (priority_score or 0.5) * 1.5  # 50% boost
                adjusted_priority = min(1.0, adjusted_priority)
            else:
                adjusted_priority = (priority_score or 0.5) * 0.5  # 50% reduction

            # Check if context exists
            existing = self.conn.execute(
                text("SELECT id FROM priority_contexts WHERE email_id = :email_id"),
                {"email_id": email_id},
            ).fetchone()

            params: dict[str, Any] = {
                "email_id": email_id,
                "sender_email": from_email,
                "sender_frequency": float(emails_from or 0),
                "sender_importance": 1.0 if is_real_person else 0.0,
                "sender_reply_rate": float(reply_rate) if reply_rate else None,
                "thread_id": str(thread_id) if thread_id else None,
                "thread_length": thread_length,
                "email_timestamp": date_parsed,
                "is_business_hours": is_business_hours,
                "age_hours": age_hours,
                "people_score": people_score,
                "temporal_score": temporal_score,
                "relationship_score": relationship_score,
                "overall_priority": adjusted_priority,
                "computed_at": now,
                "user_id": str(self.user_id) if self.user_id else None,
            }

            if existing:
                try:
                    self.conn.execute(
                        text(
                            """
                            UPDATE priority_contexts SET
                                sender_email = :sender_email,
                                sender_frequency = :sender_frequency,
                                sender_importance = :sender_importance,
                                sender_reply_rate = :sender_reply_rate,
                                thread_id = :thread_id,
                                thread_length = :thread_length,
                                email_timestamp = :email_timestamp,
                                is_business_hours = :is_business_hours,
                                age_hours = :age_hours,
                                people_score = :people_score,
                                temporal_score = :temporal_score,
                                relationship_score = :relationship_score,
                                overall_priority = :overall_priority,
                                computed_at = :computed_at
                            WHERE email_id = :email_id
                        """
                        ),
                        params,
                    )
                    updated += 1
                except Exception as e:
                    errors.append(f"Error updating context for email {email_id}: {e}")
            else:
                try:
                    self.conn.execute(
                        text(
                            """
                            INSERT INTO priority_contexts (
                                email_id, sender_email, sender_frequency,
                                sender_importance, sender_reply_rate, thread_id,
                                thread_length, email_timestamp, is_business_hours,
                                age_hours, people_score, temporal_score,
                                relationship_score, overall_priority, computed_at,
                                user_id
                            ) VALUES (
                                :email_id, :sender_email, :sender_frequency,
                                :sender_importance, :sender_reply_rate, :thread_id,
                                :thread_length, :email_timestamp, :is_business_hours,
                                :age_hours, :people_score, :temporal_score,
                                :relationship_score, :overall_priority, :computed_at,
                                :user_id
                            )
                        """
                        ),
                        params,
                    )
                    created += 1
                except Exception as e:
                    errors.append(f"Error creating context for email {email_id}: {e}")

        return created, updated, errors


def extract_all_entities(conn: Connection, user_id: uuid.UUID | None = None) -> ExtractionResult:
    """Run all entity extraction with real person filtering.

    Args:
        conn: Database connection.
        user_id: Optional user ID for multi-tenant mode.

    Returns:
        ExtractionResult with counts and errors.
    """
    all_errors: list[str] = []
    total_filtered_tasks = 0
    total_filtered_projects = 0

    # Extract tasks (real people only)
    task_extractor = TaskExtractor(conn, user_id)
    (
        llm_created,
        llm_updated,
        llm_filtered,
        llm_errors,
    ) = task_extractor.extract_from_llm_classifications()
    (
        ai_created,
        ai_updated,
        ai_filtered,
        ai_errors,
    ) = task_extractor.extract_from_ai_classifications()
    all_errors.extend(llm_errors)
    all_errors.extend(ai_errors)
    total_filtered_tasks = llm_filtered + ai_filtered

    # Extract projects (real person threads + majority real person clusters)
    project_extractor = ProjectExtractor(conn, user_id)
    (
        thread_created,
        thread_updated,
        thread_filtered,
        thread_errors,
    ) = project_extractor.extract_from_real_person_threads()
    (
        cluster_created,
        cluster_updated,
        cluster_filtered,
        cluster_errors,
    ) = project_extractor.extract_from_content_clusters()
    all_errors.extend(thread_errors)
    all_errors.extend(cluster_errors)
    total_filtered_projects = thread_filtered + cluster_filtered

    # Build priority contexts (with real person weighting)
    context_builder = PriorityContextBuilder(conn, user_id)
    ctx_created, ctx_updated, ctx_errors = context_builder.build_contexts()
    all_errors.extend(ctx_errors)

    return ExtractionResult(
        tasks_created=llm_created + ai_created,
        tasks_updated=llm_updated + ai_updated,
        projects_created=thread_created + cluster_created,
        projects_updated=thread_updated + cluster_updated,
        priority_contexts_created=ctx_created,
        priority_contexts_updated=ctx_updated,
        errors=all_errors,
        tasks_filtered_marketing=total_filtered_tasks,
        projects_filtered_marketing=total_filtered_projects,
    )
