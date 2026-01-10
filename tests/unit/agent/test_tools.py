"""Tests for agent tools."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from priority_lens.agent.context import AgentContext
from priority_lens.agent.tools import (
    PRIORITY_LENS_TOOLS,
    TOOL_EXECUTORS,
    execute_get_priority_inbox,
    execute_get_projects,
    execute_get_tasks,
    execute_search_emails,
    execute_snooze_task,
    get_priority_inbox,
    get_projects,
    get_tasks,
    search_emails,
    snooze_task,
)
from priority_lens.schemas.inbox import (
    EmailSummary,
    PriorityEmail,
    PriorityInboxResponse,
)
from priority_lens.schemas.project import ProjectListResponse, ProjectResponse
from priority_lens.schemas.task import TaskDetailResponse, TaskListResponse, TaskResponse


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def agent_context(mock_session: MagicMock) -> AgentContext:
    """Create an agent context with mock session."""

    async def factory() -> AsyncGenerator[MagicMock, None]:
        yield mock_session

    return AgentContext(
        user_id=uuid4(),
        org_id=uuid4(),
        thread_id=uuid4(),
        session_factory=factory,
    )


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_tools_list_exists(self) -> None:
        """Test that PRIORITY_LENS_TOOLS list exists."""
        assert len(PRIORITY_LENS_TOOLS) == 5

    def test_get_priority_inbox_is_tool(self) -> None:
        """Test that get_priority_inbox is a valid tool."""
        assert get_priority_inbox.name == "get_priority_inbox"
        assert "inbox" in get_priority_inbox.description.lower()

    def test_get_projects_is_tool(self) -> None:
        """Test that get_projects is a valid tool."""
        assert get_projects.name == "get_projects"
        assert "project" in get_projects.description.lower()

    def test_get_tasks_is_tool(self) -> None:
        """Test that get_tasks is a valid tool."""
        assert get_tasks.name == "get_tasks"
        assert "task" in get_tasks.description.lower()

    def test_search_emails_is_tool(self) -> None:
        """Test that search_emails is a valid tool."""
        assert search_emails.name == "search_emails"
        assert "search" in search_emails.description.lower()

    def test_snooze_task_is_tool(self) -> None:
        """Test that snooze_task is a valid tool."""
        assert snooze_task.name == "snooze_task"
        assert "snooze" in snooze_task.description.lower()


class TestToolExecutors:
    """Tests for tool executor functions."""

    def test_all_tools_have_executors(self) -> None:
        """Test that all tools have corresponding executors."""
        for tool in PRIORITY_LENS_TOOLS:
            assert tool.name in TOOL_EXECUTORS

    @pytest.mark.asyncio
    async def test_execute_get_priority_inbox(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test executing get_priority_inbox."""
        mock_response = PriorityInboxResponse(
            emails=[
                PriorityEmail(
                    email=EmailSummary(
                        id=1,
                        message_id="msg1",
                        thread_id="thread1",
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
            ],
            total=1,
            limit=10,
            offset=0,
            has_more=False,
            pending_tasks=0,
            urgent_count=0,
            from_real_people_count=0,
        )

        with patch("priority_lens.agent.tools.InboxService") as MockInboxService:
            mock_service = MockInboxService.return_value
            mock_service.get_priority_inbox = AsyncMock(return_value=mock_response)

            result = await execute_get_priority_inbox(agent_context, limit=10)

            assert len(result) == 1
            assert result[0].priority_rank == 1

    @pytest.mark.asyncio
    async def test_execute_get_priority_inbox_caps_limit(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test that limit is capped at 50."""
        mock_response = PriorityInboxResponse(
            emails=[],
            total=0,
            limit=50,
            offset=0,
            has_more=False,
            pending_tasks=0,
            urgent_count=0,
            from_real_people_count=0,
        )

        with patch("priority_lens.agent.tools.InboxService") as MockInboxService:
            mock_service = MockInboxService.return_value
            mock_service.get_priority_inbox = AsyncMock(return_value=mock_response)

            await execute_get_priority_inbox(agent_context, limit=100)

            # Verify service was called with capped limit
            mock_service.get_priority_inbox.assert_called_once()
            call_args = mock_service.get_priority_inbox.call_args
            assert call_args[1]["limit"] == 50

    @pytest.mark.asyncio
    async def test_execute_get_projects(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test executing get_projects."""
        mock_response = ProjectListResponse(
            projects=[
                ProjectResponse(
                    id=1,
                    name="Test Project",
                    project_type="software",
                    is_active=True,
                    priority=1,
                    email_count=5,
                    created_at=datetime.now(UTC),
                )
            ],
            total=1,
            limit=20,
            offset=0,
            has_more=False,
        )

        with patch("priority_lens.agent.tools.ProjectService") as MockProjectService:
            mock_service = MockProjectService.return_value
            mock_service.list_projects = AsyncMock(return_value=mock_response)

            result = await execute_get_projects(agent_context, is_active=True, limit=20)

            assert len(result) == 1
            assert result[0].name == "Test Project"

    @pytest.mark.asyncio
    async def test_execute_get_tasks(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test executing get_tasks."""
        mock_response = TaskListResponse(
            tasks=[
                TaskResponse(
                    id=1,
                    task_id="task-1",
                    description="Test task",
                    task_type="action",
                    status="pending",
                    created_at=datetime.now(UTC),
                )
            ],
            total=1,
            limit=20,
            offset=0,
            has_more=False,
        )

        with patch("priority_lens.agent.tools.TaskService") as MockTaskService:
            mock_service = MockTaskService.return_value
            mock_service.list_tasks = AsyncMock(return_value=mock_response)

            result = await execute_get_tasks(
                agent_context, status="pending", project_id=None, limit=20
            )

            assert len(result) == 1
            assert result[0].description == "Test task"

    @pytest.mark.asyncio
    async def test_execute_search_emails(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test executing search_emails (placeholder)."""
        mock_response = PriorityInboxResponse(
            emails=[],
            total=0,
            limit=10,
            offset=0,
            has_more=False,
            pending_tasks=0,
            urgent_count=0,
            from_real_people_count=0,
        )

        with patch("priority_lens.agent.tools.InboxService") as MockInboxService:
            mock_service = MockInboxService.return_value
            mock_service.get_priority_inbox = AsyncMock(return_value=mock_response)

            result = await execute_search_emails(agent_context, query="test query", limit=10)

            assert result == []

    @pytest.mark.asyncio
    async def test_execute_snooze_task(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test executing snooze_task."""
        mock_task = TaskDetailResponse(
            id=1,
            task_id="task-1",
            description="Test task",
            task_type="action",
            status="pending",
            created_at=datetime.now(UTC),
        )

        with patch("priority_lens.agent.tools.TaskService") as MockTaskService:
            mock_service = MockTaskService.return_value
            mock_service.get_task = AsyncMock(return_value=mock_task)

            result = await execute_snooze_task(
                agent_context, task_id=1, snooze_until="2025-01-15T10:00:00Z"
            )

            assert result.id == 1

    @pytest.mark.asyncio
    async def test_execute_snooze_task_invalid_datetime(self, agent_context: AgentContext) -> None:
        """Test snooze_task with invalid datetime."""
        with pytest.raises(ValueError, match="Invalid datetime format"):
            await execute_snooze_task(agent_context, task_id=1, snooze_until="not-a-date")
