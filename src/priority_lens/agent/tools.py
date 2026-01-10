"""Priority Lens tools for agent."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from priority_lens.schemas.inbox import PriorityEmail
from priority_lens.schemas.project import ProjectResponse
from priority_lens.schemas.task import TaskResponse
from priority_lens.services.inbox_service import InboxService
from priority_lens.services.project_service import ProjectService
from priority_lens.services.task_service import TaskService

if TYPE_CHECKING:
    from priority_lens.agent.context import AgentContext


@tool
async def get_priority_inbox(
    limit: int = 10,
) -> list[dict[str, object]]:
    """Get the user's priority inbox with ML-ranked emails.

    Args:
        limit: Maximum number of emails to return (default 10, max 50).

    Returns:
        List of priority-ranked email summaries with context.
    """
    # This is a placeholder - actual context injection happens in graph
    # The tool signature is for LangGraph schema generation
    return []


@tool
async def get_projects(
    is_active: bool = True,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Get the user's active projects.

    Args:
        is_active: Filter to active projects only (default True).
        limit: Maximum number of projects to return (default 20, max 50).

    Returns:
        List of project summaries.
    """
    return []


@tool
async def get_tasks(
    status: str = "pending",
    project_id: int | None = None,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Get pending tasks for the user.

    Args:
        status: Filter by status (default 'pending', options: pending, completed, dismissed).
        project_id: Optional project ID to filter by.
        limit: Maximum number of tasks to return (default 20, max 50).

    Returns:
        List of task summaries.
    """
    return []


@tool
async def search_emails(
    query: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Search emails by keyword query.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10, max 50).

    Returns:
        List of matching email summaries.
    """
    return []


@tool
async def snooze_task(
    task_id: int,
    snooze_until: str,
) -> dict[str, object]:
    """Snooze a task until a specified time.

    Args:
        task_id: The ID of the task to snooze.
        snooze_until: ISO 8601 datetime string for when to unsnooze.

    Returns:
        Updated task status.
    """
    return {}


@tool
async def generate_ui(
    ui_type: str,
    limit: int = 10,
) -> dict[str, object]:
    """Generate a server-driven UI component for the mobile client.

    Args:
        ui_type: Type of UI to generate. Options: 'inbox_list', 'task_list', 'project_list'.
        limit: Maximum number of items (default 10, max 20).

    Returns:
        UIBlock definition that the client can render.
    """
    return {}


# Tool registry for easy access
PRIORITY_LENS_TOOLS = [
    get_priority_inbox,
    get_projects,
    get_tasks,
    search_emails,
    snooze_task,
    generate_ui,
]


# Tool implementation functions (called by graph with context)


async def execute_get_priority_inbox(
    ctx: AgentContext,
    limit: int = 10,
) -> list[PriorityEmail]:
    """Execute get_priority_inbox with context.

    Args:
        ctx: Agent context with user/session info.
        limit: Maximum number of emails.

    Returns:
        List of priority emails.
    """
    limit = min(limit, 50)  # Cap at 50
    async with ctx.session() as session:
        service = InboxService(session)
        response = await service.get_priority_inbox(ctx.user_id, limit=limit)
        return response.emails


async def execute_get_projects(
    ctx: AgentContext,
    is_active: bool = True,
    limit: int = 20,
) -> list[ProjectResponse]:
    """Execute get_projects with context.

    Args:
        ctx: Agent context with user/session info.
        is_active: Filter by active status.
        limit: Maximum number of projects.

    Returns:
        List of projects.
    """
    limit = min(limit, 50)
    async with ctx.session() as session:
        service = ProjectService(session)
        response = await service.list_projects(ctx.user_id, is_active=is_active, limit=limit)
        return response.projects


async def execute_get_tasks(
    ctx: AgentContext,
    status: str = "pending",
    project_id: int | None = None,
    limit: int = 20,
) -> list[TaskResponse]:
    """Execute get_tasks with context.

    Args:
        ctx: Agent context with user/session info.
        status: Task status filter.
        project_id: Optional project filter.
        limit: Maximum number of tasks.

    Returns:
        List of tasks.
    """
    limit = min(limit, 50)
    async with ctx.session() as session:
        service = TaskService(session)
        response = await service.list_tasks(
            ctx.user_id, status=status, project_id=project_id, limit=limit
        )
        return response.tasks


async def execute_search_emails(
    ctx: AgentContext,
    query: str,
    limit: int = 10,
) -> list[PriorityEmail]:
    """Execute search_emails with context.

    Currently returns priority inbox as placeholder.

    Args:
        ctx: Agent context with user/session info.
        query: Search query.
        limit: Maximum number of results.

    Returns:
        List of matching emails.
    """
    # TODO: Implement actual email search
    limit = min(limit, 50)
    async with ctx.session() as session:
        service = InboxService(session)
        response = await service.get_priority_inbox(ctx.user_id, limit=limit)
        return response.emails


async def execute_snooze_task(
    ctx: AgentContext,
    task_id: int,
    snooze_until: str,
) -> TaskResponse:
    """Execute snooze_task with context.

    Args:
        ctx: Agent context with user/session info.
        task_id: Task to snooze.
        snooze_until: ISO datetime for unsnooze.

    Returns:
        Updated task.

    Raises:
        ValueError: If snooze_until is not a valid datetime.
    """
    try:
        # Validate datetime format (variable used for validation only)
        datetime.fromisoformat(snooze_until.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {snooze_until}") from e

    # For now, just return the task unchanged
    # TODO: Add snooze field to task model
    async with ctx.session() as session:
        service = TaskService(session)
        return await service.get_task(task_id, ctx.user_id)


async def execute_generate_ui(
    ctx: AgentContext,
    ui_type: str,
    limit: int = 10,
) -> dict[str, object]:
    """Execute generate_ui with context.

    Args:
        ctx: Agent context with user/session info.
        ui_type: Type of UI to generate.
        limit: Maximum number of items.

    Returns:
        UIBlock as a dictionary.

    Raises:
        ValueError: If ui_type is not recognized.
    """
    from priority_lens.sdui.components import (
        create_inbox_list,
        create_project_list,
        create_task_list,
    )

    limit = min(limit, 20)  # Cap at 20 for UI

    async with ctx.session() as session:
        if ui_type == "inbox_list":
            inbox_service = InboxService(session)
            inbox_response = await inbox_service.get_priority_inbox(ctx.user_id, limit=limit)
            block = create_inbox_list(inbox_response.emails)
            return block.model_dump()

        elif ui_type == "task_list":
            task_service = TaskService(session)
            task_response = await task_service.list_tasks(
                ctx.user_id, status="pending", limit=limit
            )
            block = create_task_list(task_response.tasks)
            return block.model_dump()

        elif ui_type == "project_list":
            project_service = ProjectService(session)
            project_response = await project_service.list_projects(
                ctx.user_id, is_active=True, limit=limit
            )
            block = create_project_list(project_response.projects)
            return block.model_dump()

        else:
            raise ValueError(
                f"Unknown ui_type: {ui_type}. Options: inbox_list, task_list, project_list"
            )


# Mapping from tool name to executor
TOOL_EXECUTORS = {
    "get_priority_inbox": execute_get_priority_inbox,
    "get_projects": execute_get_projects,
    "get_tasks": execute_get_tasks,
    "search_emails": execute_search_emails,
    "snooze_task": execute_snooze_task,
    "generate_ui": execute_generate_ui,
}
