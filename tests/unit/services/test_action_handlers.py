"""Tests for action handlers service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from priority_lens.sdui.schemas import ActionType
from priority_lens.services.action_handlers import (
    ActionContext,
    ActionNotFoundError,
    ActionRegistry,
    ActionResult,
    ActionResultStatus,
    ActionService,
    action_registry,
    handle_archive,
    handle_complete_task,
    handle_delete,
    handle_dismiss_task,
    handle_navigate,
    handle_reply,
    handle_snooze,
)


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def action_context() -> ActionContext:
    """Create an action context for tests."""
    return ActionContext(
        thread_id=uuid4(),
        org_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        action_id="test-action-123",
    )


class TestActionContext:
    """Tests for ActionContext dataclass."""

    def test_action_context_creation(self) -> None:
        """Test creating an action context."""
        thread_id = uuid4()
        org_id = uuid4()
        user_id = uuid4()

        ctx = ActionContext(
            thread_id=thread_id,
            org_id=org_id,
            user_id=user_id,
        )

        assert ctx.thread_id == thread_id
        assert ctx.org_id == org_id
        assert ctx.user_id == user_id
        assert ctx.session_id is None
        assert ctx.action_id is None

    def test_action_context_with_optional_fields(self) -> None:
        """Test action context with optional fields."""
        session_id = uuid4()
        ctx = ActionContext(
            thread_id=uuid4(),
            org_id=uuid4(),
            user_id=uuid4(),
            session_id=session_id,
            action_id="action-123",
        )

        assert ctx.session_id == session_id
        assert ctx.action_id == "action-123"


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_success(self) -> None:
        """Test creating a success result."""
        result = ActionResult(
            status=ActionResultStatus.SUCCESS,
            message="Task completed",
            data={"task_id": 123},
        )

        assert result.status == ActionResultStatus.SUCCESS
        assert result.message == "Task completed"
        assert result.data == {"task_id": 123}
        assert result.error is None

    def test_action_result_failure(self) -> None:
        """Test creating a failure result."""
        result = ActionResult(
            status=ActionResultStatus.FAILURE,
            message="Task not found",
            error="NOT_FOUND",
        )

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "NOT_FOUND"


class TestActionRegistry:
    """Tests for ActionRegistry class."""

    def test_register_handler(self) -> None:
        """Test registering an action handler."""
        registry = ActionRegistry()

        async def test_handler(
            ctx: ActionContext, payload: dict, session: MagicMock
        ) -> ActionResult:
            return ActionResult(status=ActionResultStatus.SUCCESS, message="OK")

        registry.register("test_action", test_handler)

        assert registry.has("test_action")
        assert registry.get("test_action") == test_handler

    def test_get_unregistered_handler(self) -> None:
        """Test getting an unregistered handler returns None."""
        registry = ActionRegistry()

        assert registry.get("nonexistent") is None
        assert not registry.has("nonexistent")

    def test_registered_types(self) -> None:
        """Test getting registered types."""
        registry = ActionRegistry()

        async def handler1(ctx: ActionContext, payload: dict, session: MagicMock) -> ActionResult:
            return ActionResult(status=ActionResultStatus.SUCCESS, message="OK")

        async def handler2(ctx: ActionContext, payload: dict, session: MagicMock) -> ActionResult:
            return ActionResult(status=ActionResultStatus.SUCCESS, message="OK")

        registry.register("type1", handler1)
        registry.register("type2", handler2)

        types = registry.registered_types
        assert "type1" in types
        assert "type2" in types


class TestGlobalActionRegistry:
    """Tests for global action registry."""

    def test_built_in_handlers_registered(self) -> None:
        """Test that built-in handlers are registered."""
        assert action_registry.has(ActionType.ARCHIVE.value)
        assert action_registry.has(ActionType.COMPLETE.value)
        assert action_registry.has(ActionType.DISMISS.value)
        assert action_registry.has(ActionType.SNOOZE.value)
        assert action_registry.has(ActionType.NAVIGATE.value)
        assert action_registry.has(ActionType.REPLY.value)
        assert action_registry.has(ActionType.DELETE.value)


class TestHandleArchive:
    """Tests for archive action handler."""

    @pytest.mark.asyncio
    async def test_archive_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful archive."""
        payload = {"email_id": 123}

        result = await handle_archive(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert "archived" in result.message.lower()
        assert result.data["email_id"] == 123

    @pytest.mark.asyncio
    async def test_archive_missing_email_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test archive with missing email_id."""
        result = await handle_archive(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"


class TestHandleCompleteTask:
    """Tests for complete task action handler."""

    @pytest.mark.asyncio
    async def test_complete_task_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful task completion."""
        payload = {"task_id": 123}

        with patch("priority_lens.services.task_service.TaskService") as MockTaskService:
            mock_service = MockTaskService.return_value
            mock_task = MagicMock()
            mock_task.status = "completed"
            mock_service.complete_task = AsyncMock(return_value=mock_task)

            result = await handle_complete_task(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["task_id"] == 123

    @pytest.mark.asyncio
    async def test_complete_task_missing_task_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test complete task with missing task_id."""
        result = await handle_complete_task(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"

    @pytest.mark.asyncio
    async def test_complete_task_not_found(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test complete task when task not found."""
        from priority_lens.services.task_service import TaskNotFoundError

        payload = {"task_id": 999}

        with patch("priority_lens.services.task_service.TaskService") as MockTaskService:
            mock_service = MockTaskService.return_value
            mock_service.complete_task = AsyncMock(side_effect=TaskNotFoundError(999))

            result = await handle_complete_task(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "NOT_FOUND"


class TestHandleDismissTask:
    """Tests for dismiss task action handler."""

    @pytest.mark.asyncio
    async def test_dismiss_task_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful task dismissal."""
        payload = {"task_id": 123}

        with patch("priority_lens.services.task_service.TaskService") as MockTaskService:
            mock_service = MockTaskService.return_value
            mock_task = MagicMock()
            mock_task.status = "dismissed"
            mock_service.dismiss_task = AsyncMock(return_value=mock_task)

            result = await handle_dismiss_task(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["task_id"] == 123

    @pytest.mark.asyncio
    async def test_dismiss_task_missing_task_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test dismiss task with missing task_id."""
        result = await handle_dismiss_task(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"


class TestHandleSnooze:
    """Tests for snooze action handler."""

    @pytest.mark.asyncio
    async def test_snooze_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful snooze."""
        payload = {
            "item_id": 123,
            "item_type": "task",
            "snooze_until": "2025-01-15T10:00:00Z",
        }

        result = await handle_snooze(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["item_id"] == 123
        assert result.data["item_type"] == "task"

    @pytest.mark.asyncio
    async def test_snooze_with_email_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test snooze with email_id instead of item_id."""
        payload = {"email_id": 456}

        result = await handle_snooze(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["item_id"] == 456

    @pytest.mark.asyncio
    async def test_snooze_missing_item_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test snooze with missing item_id."""
        result = await handle_snooze(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"


class TestHandleNavigate:
    """Tests for navigate action handler."""

    @pytest.mark.asyncio
    async def test_navigate_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful navigation."""
        payload = {"route": "/projects/123"}

        result = await handle_navigate(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["route"] == "/projects/123"

    @pytest.mark.asyncio
    async def test_navigate_missing_route(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test navigate with missing route."""
        result = await handle_navigate(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"


class TestHandleReply:
    """Tests for reply action handler."""

    @pytest.mark.asyncio
    async def test_reply_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful reply."""
        payload = {"email_id": 123}

        result = await handle_reply(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["email_id"] == 123
        assert result.data["action"] == "open_composer"

    @pytest.mark.asyncio
    async def test_reply_missing_email_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test reply with missing email_id."""
        result = await handle_reply(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"


class TestHandleDelete:
    """Tests for delete action handler."""

    @pytest.mark.asyncio
    async def test_delete_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful delete."""
        payload = {"item_id": 123}

        result = await handle_delete(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["item_id"] == 123

    @pytest.mark.asyncio
    async def test_delete_with_email_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test delete with email_id."""
        payload = {"email_id": 456}

        result = await handle_delete(action_context, payload, mock_session)

        assert result.status == ActionResultStatus.SUCCESS
        assert result.data["item_id"] == 456

    @pytest.mark.asyncio
    async def test_delete_missing_item_id(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test delete with missing item_id."""
        result = await handle_delete(action_context, {}, mock_session)

        assert result.status == ActionResultStatus.FAILURE
        assert result.error == "MISSING_PARAM"


class TestActionService:
    """Tests for ActionService class."""

    @pytest.mark.asyncio
    async def test_execute_action_success(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test successful action execution."""
        with patch("priority_lens.services.action_handlers.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = ActionService(mock_session)
            result = await service.execute_action(
                ctx=action_context,
                action_type=ActionType.ARCHIVE.value,
                payload={"email_id": 123},
            )

            assert result.status == ActionResultStatus.SUCCESS

            # Verify event was emitted
            mock_repo.append_event_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_action_not_found(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test action execution with unknown type."""
        service = ActionService(mock_session)

        with pytest.raises(ActionNotFoundError) as exc_info:
            await service.execute_action(
                ctx=action_context,
                action_type="unknown_action",
                payload={},
            )

        assert exc_info.value.action_type == "unknown_action"

    @pytest.mark.asyncio
    async def test_execute_action_without_event(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test action execution without emitting event."""
        with patch("priority_lens.services.action_handlers.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            service = ActionService(mock_session)
            result = await service.execute_action(
                ctx=action_context,
                action_type=ActionType.NAVIGATE.value,
                payload={"route": "/home"},
                emit_event=False,
            )

            assert result.status == ActionResultStatus.SUCCESS

            # Verify event was NOT emitted
            mock_repo.append_event_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_action_handles_exception(
        self,
        action_context: ActionContext,
        mock_session: MagicMock,
    ) -> None:
        """Test action execution handles exceptions gracefully."""

        # Register a handler that raises an exception
        def failing_handler():
            async def handler(
                ctx: ActionContext, payload: dict, session: MagicMock
            ) -> ActionResult:
                raise ValueError("Test error")

            return handler

        with patch("priority_lens.services.action_handlers.EventRepository") as MockEventRepo:
            mock_repo = MockEventRepo.return_value
            mock_repo.append_event_raw = AsyncMock()

            action_registry.register("failing_test", failing_handler())

            service = ActionService(mock_session)
            result = await service.execute_action(
                ctx=action_context,
                action_type="failing_test",
                payload={},
            )

            assert result.status == ActionResultStatus.FAILURE
            assert result.error == "EXECUTION_ERROR"
