"""Action handlers for SDUI actions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from priority_lens.models.canonical_event import EventActor, EventType
from priority_lens.repositories.event import EventRepository
from priority_lens.sdui.schemas import ActionType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class ActionResultStatus(str, Enum):
    """Status of an action result."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


@dataclass
class ActionResult:
    """Result of executing an action."""

    status: ActionResultStatus
    message: str
    data: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ActionContext:
    """Context for action execution."""

    thread_id: UUID
    org_id: UUID
    user_id: UUID
    session_id: UUID | None = None
    action_id: str | None = None


# Type alias for action handlers
ActionHandler = Callable[[ActionContext, dict[str, Any], "AsyncSession"], Awaitable[ActionResult]]


class ActionNotFoundError(Exception):
    """Raised when an action handler is not found."""

    def __init__(self, action_type: str) -> None:
        self.action_type = action_type
        super().__init__(f"No handler found for action type: {action_type}")


class ActionExecutionError(Exception):
    """Raised when action execution fails."""

    def __init__(self, action_type: str, message: str) -> None:
        self.action_type = action_type
        self.message = message
        super().__init__(f"Action {action_type} failed: {message}")


# ============================================================================
# Action Handler Registry
# ============================================================================


class ActionRegistry:
    """Registry for action handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, action_type: str, handler: ActionHandler) -> None:
        """Register an action handler.

        Args:
            action_type: The action type to handle.
            handler: The handler function.
        """
        self._handlers[action_type] = handler

    def get(self, action_type: str) -> ActionHandler | None:
        """Get an action handler.

        Args:
            action_type: The action type.

        Returns:
            Handler function or None if not found.
        """
        return self._handlers.get(action_type)

    def has(self, action_type: str) -> bool:
        """Check if an action handler exists.

        Args:
            action_type: The action type.

        Returns:
            True if handler exists.
        """
        return action_type in self._handlers

    @property
    def registered_types(self) -> list[str]:
        """Get list of registered action types."""
        return list(self._handlers.keys())


# Global action registry
action_registry = ActionRegistry()


# ============================================================================
# Built-in Action Handlers
# ============================================================================


async def handle_archive(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle archive email action.

    Args:
        ctx: Action context.
        payload: Action payload with email_id.
        session: Database session.

    Returns:
        Action result.
    """
    email_id = payload.get("email_id")
    if not email_id:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing email_id in payload",
            error="MISSING_PARAM",
        )

    # In a real implementation, this would call the email service
    await logger.ainfo(
        "action_archive_email",
        email_id=email_id,
        user_id=str(ctx.user_id),
    )

    return ActionResult(
        status=ActionResultStatus.SUCCESS,
        message=f"Email {email_id} archived",
        data={"email_id": email_id},
    )


async def handle_complete_task(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle complete task action.

    Args:
        ctx: Action context.
        payload: Action payload with task_id.
        session: Database session.

    Returns:
        Action result.
    """
    from priority_lens.services.task_service import TaskNotFoundError, TaskService

    task_id = payload.get("task_id")
    if not task_id:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing task_id in payload",
            error="MISSING_PARAM",
        )

    try:
        task_service = TaskService(session)
        task = await task_service.complete_task(int(task_id), ctx.user_id)

        await logger.ainfo(
            "action_complete_task",
            task_id=task_id,
            user_id=str(ctx.user_id),
        )

        return ActionResult(
            status=ActionResultStatus.SUCCESS,
            message=f"Task {task_id} completed",
            data={"task_id": task_id, "status": task.status},
        )
    except TaskNotFoundError:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message=f"Task {task_id} not found",
            error="NOT_FOUND",
        )
    except Exception as e:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message=str(e),
            error="INTERNAL_ERROR",
        )


async def handle_dismiss_task(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle dismiss task action.

    Args:
        ctx: Action context.
        payload: Action payload with task_id.
        session: Database session.

    Returns:
        Action result.
    """
    from priority_lens.services.task_service import TaskNotFoundError, TaskService

    task_id = payload.get("task_id")
    if not task_id:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing task_id in payload",
            error="MISSING_PARAM",
        )

    try:
        task_service = TaskService(session)
        task = await task_service.dismiss_task(int(task_id), ctx.user_id)

        await logger.ainfo(
            "action_dismiss_task",
            task_id=task_id,
            user_id=str(ctx.user_id),
        )

        return ActionResult(
            status=ActionResultStatus.SUCCESS,
            message=f"Task {task_id} dismissed",
            data={"task_id": task_id, "status": task.status},
        )
    except TaskNotFoundError:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message=f"Task {task_id} not found",
            error="NOT_FOUND",
        )
    except Exception as e:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message=str(e),
            error="INTERNAL_ERROR",
        )


async def handle_snooze(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle snooze action (email or task).

    Args:
        ctx: Action context.
        payload: Action payload with item_id and snooze_until.
        session: Database session.

    Returns:
        Action result.
    """
    item_type = payload.get("item_type", "email")
    item_id = payload.get("item_id") or payload.get("task_id") or payload.get("email_id")
    snooze_until = payload.get("snooze_until")

    if not item_id:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing item_id in payload",
            error="MISSING_PARAM",
        )

    await logger.ainfo(
        "action_snooze",
        item_type=item_type,
        item_id=item_id,
        snooze_until=snooze_until,
        user_id=str(ctx.user_id),
    )

    return ActionResult(
        status=ActionResultStatus.SUCCESS,
        message=f"{item_type.capitalize()} {item_id} snoozed",
        data={"item_type": item_type, "item_id": item_id, "snooze_until": snooze_until},
    )


async def handle_navigate(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle navigation action.

    Args:
        ctx: Action context.
        payload: Action payload with route.
        session: Database session.

    Returns:
        Action result.
    """
    route = payload.get("route")
    if not route:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing route in payload",
            error="MISSING_PARAM",
        )

    await logger.ainfo(
        "action_navigate",
        route=route,
        user_id=str(ctx.user_id),
    )

    return ActionResult(
        status=ActionResultStatus.SUCCESS,
        message=f"Navigate to {route}",
        data={"route": route},
    )


async def handle_reply(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle reply to email action.

    Args:
        ctx: Action context.
        payload: Action payload with email_id.
        session: Database session.

    Returns:
        Action result.
    """
    email_id = payload.get("email_id")
    if not email_id:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing email_id in payload",
            error="MISSING_PARAM",
        )

    await logger.ainfo(
        "action_reply",
        email_id=email_id,
        user_id=str(ctx.user_id),
    )

    # Return navigation hint to open reply composer
    return ActionResult(
        status=ActionResultStatus.SUCCESS,
        message=f"Opening reply for email {email_id}",
        data={"email_id": email_id, "action": "open_composer"},
    )


async def handle_delete(
    ctx: ActionContext,
    payload: dict[str, Any],
    session: AsyncSession,
) -> ActionResult:
    """Handle delete action.

    Args:
        ctx: Action context.
        payload: Action payload with item_id.
        session: Database session.

    Returns:
        Action result.
    """
    item_id = payload.get("item_id") or payload.get("email_id") or payload.get("task_id")
    if not item_id:
        return ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Missing item_id in payload",
            error="MISSING_PARAM",
        )

    await logger.ainfo(
        "action_delete",
        item_id=item_id,
        user_id=str(ctx.user_id),
    )

    return ActionResult(
        status=ActionResultStatus.SUCCESS,
        message=f"Item {item_id} deleted",
        data={"item_id": item_id},
    )


# Register built-in handlers
action_registry.register(ActionType.ARCHIVE.value, handle_archive)
action_registry.register(ActionType.COMPLETE.value, handle_complete_task)
action_registry.register(ActionType.DISMISS.value, handle_dismiss_task)
action_registry.register(ActionType.SNOOZE.value, handle_snooze)
action_registry.register(ActionType.NAVIGATE.value, handle_navigate)
action_registry.register(ActionType.REPLY.value, handle_reply)
action_registry.register(ActionType.DELETE.value, handle_delete)


# ============================================================================
# Action Service
# ============================================================================


class ActionService:
    """Service for executing SDUI actions."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize action service.

        Args:
            session: Database session.
        """
        self._session = session
        self._event_repo = EventRepository(session)

    async def execute_action(
        self,
        ctx: ActionContext,
        action_type: str,
        payload: dict[str, Any],
        *,
        emit_event: bool = True,
    ) -> ActionResult:
        """Execute an action and optionally emit result event.

        Args:
            ctx: Action context with thread/user info.
            action_type: Type of action to execute.
            payload: Action payload.
            emit_event: Whether to emit ui.action.result event.

        Returns:
            Action result.

        Raises:
            ActionNotFoundError: If action handler not found.
        """
        handler = action_registry.get(action_type)
        if handler is None:
            raise ActionNotFoundError(action_type)

        await logger.ainfo(
            "action_executing",
            action_type=action_type,
            action_id=ctx.action_id,
            thread_id=str(ctx.thread_id),
        )

        try:
            result = await handler(ctx, payload, self._session)
        except Exception as e:
            result = ActionResult(
                status=ActionResultStatus.FAILURE,
                message=str(e),
                error="EXECUTION_ERROR",
            )
            await logger.aerror(
                "action_failed",
                action_type=action_type,
                error=str(e),
            )

        # Emit result event if requested
        if emit_event:
            await self._emit_result_event(ctx, action_type, result)

        await logger.ainfo(
            "action_completed",
            action_type=action_type,
            status=result.status.value,
        )

        return result

    async def _emit_result_event(
        self,
        ctx: ActionContext,
        action_type: str,
        result: ActionResult,
    ) -> None:
        """Emit action result event.

        Args:
            ctx: Action context.
            action_type: Type of action executed.
            result: Action result.
        """
        await self._event_repo.append_event_raw(
            thread_id=ctx.thread_id,
            org_id=ctx.org_id,
            actor=EventActor.SYSTEM,
            event_type=EventType.UI_ACTION_RESULT,
            payload={
                "action_id": ctx.action_id,
                "action_type": action_type,
                "status": result.status.value,
                "message": result.message,
                "data": result.data,
                "error": result.error,
            },
            session_id=ctx.session_id,
            user_id=ctx.user_id,
        )
