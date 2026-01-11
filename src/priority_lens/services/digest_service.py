"""Digest service for the Smart Digest feature."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from priority_lens.schemas.digest import (
    DigestAction,
    DigestResponse,
    DigestTodoItem,
    DigestTopicItem,
    UrgencyLevel,
)


class DigestService:
    """Service for generating personalized daily digests."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize service with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    async def get_digest(
        self,
        user_id: UUID | None = None,
        *,
        max_todos: int = 5,
        max_topics: int = 5,
        user_name: str | None = None,
    ) -> DigestResponse:
        """Generate personalized daily digest.

        Args:
            user_id: User UUID (for multi-tenant support).
            max_todos: Maximum number of to-do items to include.
            max_topics: Maximum number of topics to include.
            user_name: User's name for personalized greeting.

        Returns:
            Complete digest response with greeting, todos, and topics.
        """
        # Generate greeting based on time of day
        greeting = self._generate_greeting(user_name)

        # Get actionable to-dos
        todos = await self._get_actionable_todos(user_id, max_todos)

        # Get topics to catch up on
        topics = await self._get_topics_to_catchup(user_id, max_topics)

        # Generate subtitle
        urgent_count = sum(1 for t in todos if t.urgency == UrgencyLevel.HIGH)
        if urgent_count > 0:
            subtitle = f"{urgent_count} urgent item{'s' if urgent_count > 1 else ''} need{'s' if urgent_count == 1 else ''} attention"
        elif len(todos) > 0:
            subtitle = f"{len(todos)} item{'s' if len(todos) > 1 else ''} to review"
        else:
            subtitle = "All caught up!"

        return DigestResponse(
            greeting=greeting,
            subtitle=subtitle,
            suggested_todos=todos,
            topics_to_catchup=topics,
            last_updated=datetime.now(UTC),
        )

    def _generate_greeting(self, user_name: str | None = None) -> str:
        """Generate time-appropriate greeting.

        Args:
            user_name: User's name for personalization.

        Returns:
            Personalized greeting string.
        """
        hour = datetime.now(UTC).hour

        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Hello"

        if user_name:
            return f"{time_greeting}, {user_name}"
        return time_greeting

    async def _get_actionable_todos(
        self,
        user_id: UUID | None,
        limit: int,
    ) -> list[DigestTodoItem]:
        """Get actionable to-do items from emails and tasks.

        Prioritizes:
        - Emails requiring replies (action = 'received' with no reply)
        - High priority emails from real people
        - Tasks with upcoming due dates

        Args:
            user_id: User UUID for scoping.
            limit: Maximum items to return.

        Returns:
            List of actionable to-do items.
        """
        # Get emails that need action
        query = text(
            """
            WITH urgent_emails AS (
                SELECT
                    e.id,
                    e.message_id,
                    e.subject,
                    e.from_email,
                    e.from_name,
                    e.date_parsed,
                    e.body_preview,
                    ep.priority_score,
                    ep.priority_rank,
                    pc.sender_importance,
                    pc.age_hours
                FROM emails e
                LEFT JOIN email_priority ep ON ep.email_id = e.id
                LEFT JOIN priority_contexts pc ON pc.email_id = e.id
                WHERE e.is_sent = false
                AND e.action = 'received'
                AND e.date_parsed > NOW() - INTERVAL '7 days'
                ORDER BY ep.priority_rank ASC NULLS LAST
                LIMIT :limit
            )
            SELECT * FROM urgent_emails
            """
        )

        result = await self._session.execute(query, {"limit": limit})
        rows = result.fetchall()

        todos: list[DigestTodoItem] = []
        for row in rows:
            email_id = row[0]
            subject = row[2] or "No subject"
            from_email = row[3] or ""
            from_name = row[4]
            priority_score = row[7] or 0.0
            age_hours = row[10] or 0

            # Determine urgency based on priority score and age
            if priority_score > 0.7 or age_hours > 48:
                urgency = UrgencyLevel.HIGH
            elif priority_score > 0.4:
                urgency = UrgencyLevel.MEDIUM
            else:
                urgency = UrgencyLevel.LOW

            # Determine due string
            if age_hours > 72:
                due = "Overdue"
            elif age_hours > 24:
                due = "Today"
            else:
                due = None

            # Build source string
            source = f"Email from {from_name or from_email}"

            # Build actions
            actions = [
                DigestAction(
                    id=f"reply_{email_id}",
                    type="reply",
                    label="Reply",
                    endpoint="/api/v1/actions",
                    params={"action_type": "reply", "email_id": email_id},
                ),
                DigestAction(
                    id=f"dismiss_{email_id}",
                    type="dismiss",
                    label="Dismiss",
                    endpoint="/api/v1/actions",
                    params={"action_type": "dismiss", "email_id": email_id},
                ),
            ]

            todo = DigestTodoItem(
                id=f"email_{email_id}",
                title=subject[:80] + ("..." if len(subject) > 80 else ""),
                source=source,
                urgency=urgency,
                due=due,
                context=None,
                email_id=str(email_id),
                actions=actions,
            )
            todos.append(todo)

        return todos

    async def _get_topics_to_catchup(
        self,
        user_id: UUID | None,
        limit: int,
    ) -> list[DigestTopicItem]:
        """Get topics grouped by thread or cluster.

        Groups emails by thread_id and returns clusters with multiple emails.

        Args:
            user_id: User UUID for scoping.
            limit: Maximum topics to return.

        Returns:
            List of topic items to catch up on.
        """
        # Get email threads with multiple emails
        query = text(
            """
            WITH thread_stats AS (
                SELECT
                    e.thread_id,
                    COUNT(*) as email_count,
                    MAX(e.date_parsed) as last_activity,
                    MIN(e.date_parsed) as first_activity,
                    ARRAY_AGG(DISTINCT COALESCE(e.from_name, e.from_email)) as participants,
                    STRING_AGG(DISTINCT e.subject, ', ') as subjects
                FROM emails e
                WHERE e.date_parsed > NOW() - INTERVAL '7 days'
                GROUP BY e.thread_id
                HAVING COUNT(*) > 1
                ORDER BY MAX(e.date_parsed) DESC
                LIMIT :limit
            )
            SELECT
                ts.thread_id,
                ts.email_count,
                ts.last_activity,
                ts.participants,
                ts.subjects
            FROM thread_stats ts
            """
        )

        result = await self._session.execute(query, {"limit": limit})
        rows = result.fetchall()

        topics: list[DigestTopicItem] = []
        for row in rows:
            thread_id = row[0]
            email_count = row[1]
            last_activity = row[2]
            participants = row[3] or []
            subjects = row[4] or "Thread"

            # Calculate last activity string
            if last_activity:
                age = datetime.now(UTC) - last_activity.replace(tzinfo=UTC)
                if age.total_seconds() < 3600:
                    last_activity_str = f"{int(age.total_seconds() / 60)} minutes ago"
                elif age.total_seconds() < 86400:
                    last_activity_str = f"{int(age.total_seconds() / 3600)} hours ago"
                else:
                    last_activity_str = f"{int(age.days)} days ago"
            else:
                last_activity_str = "Unknown"

            # Clean up title from subjects
            title = subjects.split(",")[0].strip()[:60]
            if len(subjects.split(",")[0].strip()) > 60:
                title += "..."

            # Determine urgency based on email count
            if email_count > 5:
                urgency = UrgencyLevel.HIGH
            elif email_count > 3:
                urgency = UrgencyLevel.MEDIUM
            else:
                urgency = UrgencyLevel.LOW

            topic = DigestTopicItem(
                id=f"thread_{thread_id}" if thread_id else f"topic_{len(topics)}",
                title=title,
                email_count=email_count,
                participants=participants[:3],  # Limit to 3 participants shown
                last_activity=last_activity_str,
                summary=None,
                urgency=urgency,
            )
            topics.append(topic)

        return topics
