"""Tests for agent graph."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from priority_lens.agent.context import AgentContext
from priority_lens.agent.graph import AgentRunner, create_agent_graph
from priority_lens.agent.state import AgentState


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


class TestCreateAgentGraph:
    """Tests for create_agent_graph."""

    def test_graph_compiles(self) -> None:
        """Test that graph compiles without errors."""
        with patch("priority_lens.agent.graph.ChatAnthropic"):
            graph = create_agent_graph()
            assert graph is not None

    def test_graph_with_custom_model(self) -> None:
        """Test graph creation with custom model."""
        with patch("priority_lens.agent.graph.ChatAnthropic") as MockChat:
            create_agent_graph(model_name="claude-3-haiku-20240307")
            MockChat.assert_called_once()
            call_kwargs = MockChat.call_args[1]
            assert call_kwargs["model_name"] == "claude-3-haiku-20240307"


class TestAgentRunner:
    """Tests for AgentRunner."""

    def test_runner_initialization(self, agent_context: AgentContext) -> None:
        """Test runner initialization."""
        with patch("priority_lens.agent.graph.create_agent_graph") as mock_create:
            mock_create.return_value = MagicMock()
            runner = AgentRunner(agent_context)

            assert runner._ctx is agent_context
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_runner_execute_tool_unknown(self, agent_context: AgentContext) -> None:
        """Test executing unknown tool raises error."""
        with patch("priority_lens.agent.graph.create_agent_graph"):
            runner = AgentRunner(agent_context)

            with pytest.raises(ValueError, match="Unknown tool"):
                await runner.execute_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_runner_execute_tool_valid(
        self, agent_context: AgentContext, mock_session: MagicMock
    ) -> None:
        """Test executing a valid tool."""
        from priority_lens.schemas.inbox import PriorityInboxResponse

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

        with (
            patch("priority_lens.agent.graph.create_agent_graph"),
            patch("priority_lens.agent.tools.InboxService") as MockInboxService,
        ):
            mock_service = MockInboxService.return_value
            mock_service.get_priority_inbox = AsyncMock(return_value=mock_response)

            runner = AgentRunner(agent_context)
            result = await runner.execute_tool("get_priority_inbox", {"limit": 5})

            assert result == []


class TestAgentState:
    """Tests for agent state handling in graph."""

    def test_state_message_accumulation(self) -> None:
        """Test that messages accumulate in state."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        state: AgentState = {"messages": messages}

        # Verify messages are stored
        assert len(state["messages"]) == 2
        assert state["messages"][0].content == "Hello"
        assert state["messages"][1].content == "Hi there!"
