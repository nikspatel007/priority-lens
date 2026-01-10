"""Tests for SDUI component factories."""

from __future__ import annotations

from datetime import UTC, datetime

from priority_lens.schemas.inbox import EmailSummary, PriorityEmail
from priority_lens.schemas.project import ProjectResponse
from priority_lens.schemas.task import TaskResponse
from priority_lens.sdui.components import (
    create_email_card,
    create_inbox_list,
    create_project_card,
    create_project_list,
    create_task_card,
    create_task_list,
)
from priority_lens.sdui.schemas import ActionType


class TestCreateEmailCard:
    """Tests for create_email_card function."""

    def test_creates_email_card(self) -> None:
        """Test creating an email card."""
        email = PriorityEmail(
            email=EmailSummary(
                id=1,
                message_id="msg-1",
                thread_id="thread-1",
                subject="Test Subject",
                from_email="sender@example.com",
                from_name="Sender Name",
                date_parsed=datetime.now(UTC),
                body_preview="This is the preview...",
                has_attachments=True,
            ),
            priority_rank=1,
            priority_score=0.95,
            context=None,
            task_count=2,
            project_name="Test Project",
        )

        block = create_email_card(email)

        assert block.id == "email-1"
        assert block.type == "email_card"
        assert block.props["subject"] == "Test Subject"
        assert block.props["from_email"] == "sender@example.com"
        assert block.props["from_name"] == "Sender Name"
        assert block.props["priority_rank"] == 1
        assert block.props["has_attachments"] is True
        assert block.props["task_count"] == 2
        assert block.props["project_name"] == "Test Project"

    def test_email_card_with_actions(self) -> None:
        """Test email card includes actions by default."""
        email = PriorityEmail(
            email=EmailSummary(
                id=1,
                message_id="msg-1",
                thread_id="thread-1",
                subject="Test",
                from_email="test@example.com",
                date_parsed=datetime.now(UTC),
            ),
            priority_rank=1,
            priority_score=0.9,
            context=None,
            task_count=0,
            project_name=None,
        )

        block = create_email_card(email, show_actions=True)

        assert len(block.actions) == 3
        action_types = [a.type for a in block.actions]
        assert ActionType.REPLY in action_types
        assert ActionType.ARCHIVE in action_types
        assert ActionType.SNOOZE in action_types

    def test_email_card_without_actions(self) -> None:
        """Test email card without actions."""
        email = PriorityEmail(
            email=EmailSummary(
                id=1,
                message_id="msg-1",
                thread_id="thread-1",
                subject="Test",
                from_email="test@example.com",
                date_parsed=datetime.now(UTC),
            ),
            priority_rank=1,
            priority_score=0.9,
            context=None,
            task_count=0,
            project_name=None,
        )

        block = create_email_card(email, show_actions=False)

        assert len(block.actions) == 0

    def test_email_card_has_layout(self) -> None:
        """Test email card has layout props."""
        email = PriorityEmail(
            email=EmailSummary(
                id=1,
                message_id="msg-1",
                thread_id="thread-1",
                subject="Test",
                from_email="test@example.com",
                date_parsed=datetime.now(UTC),
            ),
            priority_rank=1,
            priority_score=0.9,
            context=None,
            task_count=0,
            project_name=None,
        )

        block = create_email_card(email)

        assert block.layout is not None
        assert block.layout.padding == 12
        assert block.layout.border_radius == 8


class TestCreateTaskCard:
    """Tests for create_task_card function."""

    def test_creates_task_card(self) -> None:
        """Test creating a task card."""
        task = TaskResponse(
            id=1,
            task_id="task-1",
            description="Complete the report",
            task_type="action",
            complexity="medium",
            status="pending",
            urgency_score=0.8,
            deadline=datetime.now(UTC),
            deadline_text="Due tomorrow",
            assigned_to="user@example.com",
            is_assigned_to_user=True,
            created_at=datetime.now(UTC),
        )

        block = create_task_card(task)

        assert block.id == "task-1"
        assert block.type == "task_card"
        assert block.props["description"] == "Complete the report"
        assert block.props["task_type"] == "action"
        assert block.props["status"] == "pending"
        assert block.props["urgency_score"] == 0.8
        assert block.props["is_assigned_to_user"] is True

    def test_task_card_with_actions(self) -> None:
        """Test task card includes actions by default."""
        task = TaskResponse(
            id=1,
            task_id="task-1",
            description="Test",
            task_type="action",
            status="pending",
            created_at=datetime.now(UTC),
        )

        block = create_task_card(task, show_actions=True)

        assert len(block.actions) == 3
        action_types = [a.type for a in block.actions]
        assert ActionType.COMPLETE in action_types
        assert ActionType.SNOOZE in action_types
        assert ActionType.DISMISS in action_types

    def test_task_card_dismiss_has_confirmation(self) -> None:
        """Test dismiss action has confirmation message."""
        task = TaskResponse(
            id=1,
            task_id="task-1",
            description="Test",
            task_type="action",
            status="pending",
            created_at=datetime.now(UTC),
        )

        block = create_task_card(task)

        dismiss_action = next(a for a in block.actions if a.type == ActionType.DISMISS)
        assert dismiss_action.confirm is not None
        assert "dismiss" in dismiss_action.confirm.lower()


class TestCreateProjectCard:
    """Tests for create_project_card function."""

    def test_creates_project_card(self) -> None:
        """Test creating a project card."""
        project = ProjectResponse(
            id=1,
            name="Test Project",
            project_type="software",
            is_active=True,
            priority=1,
            email_count=10,
            last_activity=datetime.now(UTC),
            owner_email="owner@example.com",
            participants=["user1@example.com", "user2@example.com"],
            created_at=datetime.now(UTC),
        )

        block = create_project_card(project)

        assert block.id == "project-1"
        assert block.type == "project_card"
        assert block.props["name"] == "Test Project"
        assert block.props["project_type"] == "software"
        assert block.props["is_active"] is True
        assert block.props["email_count"] == 10
        assert block.props["owner_email"] == "owner@example.com"

    def test_project_card_with_navigation(self) -> None:
        """Test project card has navigation action."""
        project = ProjectResponse(
            id=1,
            name="Test",
            is_active=True,
            priority=1,
            email_count=0,
            created_at=datetime.now(UTC),
        )

        block = create_project_card(project, show_actions=True)

        assert len(block.actions) == 1
        assert block.actions[0].type == ActionType.NAVIGATE
        assert block.actions[0].params["route"] == "/projects/1"


class TestCreateInboxList:
    """Tests for create_inbox_list function."""

    def test_creates_inbox_list(self) -> None:
        """Test creating an inbox list."""
        emails = [
            PriorityEmail(
                email=EmailSummary(
                    id=i,
                    message_id=f"msg-{i}",
                    thread_id=f"thread-{i}",
                    subject=f"Subject {i}",
                    from_email=f"sender{i}@example.com",
                    date_parsed=datetime.now(UTC),
                ),
                priority_rank=i,
                priority_score=0.9 - (i * 0.1),
                context=None,
                task_count=0,
                project_name=None,
            )
            for i in range(3)
        ]

        block = create_inbox_list(emails, title="My Inbox")

        assert block.type == "list"
        assert block.props["title"] == "My Inbox"
        assert block.props["item_count"] == 3
        assert len(block.children) == 3

        # Verify children are email cards
        for child in block.children:
            assert child.type == "email_card"

    def test_inbox_list_empty(self) -> None:
        """Test empty inbox list."""
        block = create_inbox_list([])

        assert block.props["item_count"] == 0
        assert block.props["empty_message"] == "No priority emails"
        assert len(block.children) == 0


class TestCreateTaskList:
    """Tests for create_task_list function."""

    def test_creates_task_list(self) -> None:
        """Test creating a task list."""
        tasks = [
            TaskResponse(
                id=i,
                task_id=f"task-{i}",
                description=f"Task {i}",
                task_type="action",
                status="pending",
                created_at=datetime.now(UTC),
            )
            for i in range(2)
        ]

        block = create_task_list(tasks, title="My Tasks")

        assert block.type == "list"
        assert block.props["title"] == "My Tasks"
        assert block.props["item_count"] == 2
        assert len(block.children) == 2

        for child in block.children:
            assert child.type == "task_card"


class TestCreateProjectList:
    """Tests for create_project_list function."""

    def test_creates_project_list(self) -> None:
        """Test creating a project list."""
        projects = [
            ProjectResponse(
                id=i,
                name=f"Project {i}",
                is_active=True,
                priority=i,
                email_count=i * 5,
                created_at=datetime.now(UTC),
            )
            for i in range(2)
        ]

        block = create_project_list(projects)

        assert block.type == "list"
        assert block.props["item_count"] == 2
        assert len(block.children) == 2

        for child in block.children:
            assert child.type == "project_card"
