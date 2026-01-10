"""SDUI component factory functions."""

# mypy: disable-error-code="call-arg"
# Note: mypy doesn't understand Pydantic's Field defaults and aliases properly
# The tests verify correct runtime behavior

from __future__ import annotations

from uuid import uuid4

from priority_lens.schemas.inbox import PriorityEmail
from priority_lens.schemas.project import ProjectResponse
from priority_lens.schemas.task import TaskResponse
from priority_lens.sdui.schemas import ActionProps, ActionType, LayoutProps, UIBlock


def _generate_id() -> str:
    """Generate a unique component ID."""
    return str(uuid4())[:8]


def create_email_card(
    email: PriorityEmail,
    *,
    show_actions: bool = True,
) -> UIBlock:
    """Create an email card UI block.

    Args:
        email: Priority email data.
        show_actions: Whether to include action buttons.

    Returns:
        UIBlock for the email card.
    """
    block_id = f"email-{email.email.id}"

    # Build props from email data
    props = {
        "email_id": email.email.id,
        "subject": email.email.subject or "(No subject)",
        "from_email": email.email.from_email,
        "from_name": email.email.from_name,
        "date": email.email.date_parsed.isoformat() if email.email.date_parsed else None,
        "preview": email.email.body_preview,
        "priority_rank": email.priority_rank,
        "priority_score": email.priority_score,
        "has_attachments": email.email.has_attachments,
        "task_count": email.task_count,
        "project_name": email.project_name,
    }

    # Build actions
    actions: list[ActionProps] = []
    if show_actions:
        actions = [
            ActionProps(
                id=f"reply-{email.email.id}",
                type=ActionType.REPLY,
                label="Reply",
                icon="reply",
            ),
            ActionProps(
                id=f"archive-{email.email.id}",
                type=ActionType.ARCHIVE,
                label="Archive",
                endpoint=f"/api/emails/{email.email.id}/archive",
                method="POST",
                icon="archive",
            ),
            ActionProps(
                id=f"snooze-{email.email.id}",
                type=ActionType.SNOOZE,
                label="Snooze",
                icon="clock",
            ),
        ]

    return UIBlock(
        id=block_id,
        type="email_card",
        props=props,
        layout=LayoutProps(padding=12, border_radius=8),
        actions=actions,
    )


def create_task_card(
    task: TaskResponse,
    *,
    show_actions: bool = True,
) -> UIBlock:
    """Create a task card UI block.

    Args:
        task: Task data.
        show_actions: Whether to include action buttons.

    Returns:
        UIBlock for the task card.
    """
    block_id = f"task-{task.id}"

    props = {
        "task_id": task.id,
        "description": task.description,
        "task_type": task.task_type,
        "complexity": task.complexity,
        "status": task.status,
        "urgency_score": task.urgency_score,
        "deadline": task.deadline.isoformat() if task.deadline else None,
        "deadline_text": task.deadline_text,
        "assigned_to": task.assigned_to,
        "is_assigned_to_user": task.is_assigned_to_user,
    }

    actions: list[ActionProps] = []
    if show_actions:
        actions = [
            ActionProps(
                id=f"complete-{task.id}",
                type=ActionType.COMPLETE,
                label="Complete",
                endpoint=f"/api/tasks/{task.id}/complete",
                method="POST",
                icon="check",
            ),
            ActionProps(
                id=f"snooze-{task.id}",
                type=ActionType.SNOOZE,
                label="Snooze",
                icon="clock",
            ),
            ActionProps(
                id=f"dismiss-{task.id}",
                type=ActionType.DISMISS,
                label="Dismiss",
                endpoint=f"/api/tasks/{task.id}/dismiss",
                method="POST",
                confirm="Are you sure you want to dismiss this task?",
                icon="x",
            ),
        ]

    return UIBlock(
        id=block_id,
        type="task_card",
        props=props,
        layout=LayoutProps(padding=12, border_radius=8),
        actions=actions,
    )


def create_project_card(
    project: ProjectResponse,
    *,
    show_actions: bool = True,
) -> UIBlock:
    """Create a project card UI block.

    Args:
        project: Project data.
        show_actions: Whether to include action buttons.

    Returns:
        UIBlock for the project card.
    """
    block_id = f"project-{project.id}"

    props = {
        "project_id": project.id,
        "name": project.name,
        "project_type": project.project_type,
        "is_active": project.is_active,
        "priority": project.priority,
        "email_count": project.email_count,
        "last_activity": (project.last_activity.isoformat() if project.last_activity else None),
        "owner_email": project.owner_email,
        "participants": project.participants,
    }

    actions: list[ActionProps] = []
    if show_actions:
        actions = [
            ActionProps(
                id=f"view-{project.id}",
                type=ActionType.NAVIGATE,
                label="View Project",
                params={"route": f"/projects/{project.id}"},
                icon="folder",
            ),
        ]

    return UIBlock(
        id=block_id,
        type="project_card",
        props=props,
        layout=LayoutProps(padding=12, border_radius=8),
        actions=actions,
    )


def create_inbox_list(
    emails: list[PriorityEmail],
    *,
    title: str = "Priority Inbox",
    show_actions: bool = True,
) -> UIBlock:
    """Create an inbox list UI block containing email cards.

    Args:
        emails: List of priority emails.
        title: Title for the list.
        show_actions: Whether to include action buttons on email cards.

    Returns:
        UIBlock for the inbox list.
    """
    block_id = f"inbox-list-{_generate_id()}"

    # Create child email cards
    children = [create_email_card(email, show_actions=show_actions) for email in emails]

    return UIBlock(
        id=block_id,
        type="list",
        props={
            "title": title,
            "item_count": len(emails),
            "empty_message": "No priority emails",
        },
        layout=LayoutProps(padding=16),
        children=children,
    )


def create_task_list(
    tasks: list[TaskResponse],
    *,
    title: str = "Tasks",
    show_actions: bool = True,
) -> UIBlock:
    """Create a task list UI block containing task cards.

    Args:
        tasks: List of tasks.
        title: Title for the list.
        show_actions: Whether to include action buttons on task cards.

    Returns:
        UIBlock for the task list.
    """
    block_id = f"task-list-{_generate_id()}"

    children = [create_task_card(task, show_actions=show_actions) for task in tasks]

    return UIBlock(
        id=block_id,
        type="list",
        props={
            "title": title,
            "item_count": len(tasks),
            "empty_message": "No tasks",
        },
        layout=LayoutProps(padding=16),
        children=children,
    )


def create_project_list(
    projects: list[ProjectResponse],
    *,
    title: str = "Projects",
    show_actions: bool = True,
) -> UIBlock:
    """Create a project list UI block containing project cards.

    Args:
        projects: List of projects.
        title: Title for the list.
        show_actions: Whether to include action buttons on project cards.

    Returns:
        UIBlock for the project list.
    """
    block_id = f"project-list-{_generate_id()}"

    children = [create_project_card(project, show_actions=show_actions) for project in projects]

    return UIBlock(
        id=block_id,
        type="list",
        props={
            "title": title,
            "item_count": len(projects),
            "empty_message": "No projects",
        },
        layout=LayoutProps(padding=16),
        children=children,
    )
