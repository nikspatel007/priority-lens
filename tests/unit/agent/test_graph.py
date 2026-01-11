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

    @pytest.mark.asyncio
    async def test_runner_run_streaming_text_response(self, agent_context: AgentContext) -> None:
        """Test run_streaming yields text events from AI messages."""
        from priority_lens.models.canonical_event import EventType

        # Mock graph that returns a simple text response
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi! How can I help you?"),
                ]
            }
        )

        with patch("priority_lens.agent.graph.create_agent_graph", return_value=mock_graph):
            runner = AgentRunner(agent_context)
            events_gen = await runner.run_streaming("Hello")

            # Collect events
            events = [event async for event in events_gen]

            # Should have one text event from AI response
            assert len(events) == 1
            assert events[0].event_type == EventType.ASSISTANT_TEXT_FINAL
            assert events[0].payload["text"] == "Hi! How can I help you?"

    @pytest.mark.asyncio
    async def test_runner_run_streaming_tool_calls(self, agent_context: AgentContext) -> None:
        """Test run_streaming yields tool call and result events."""
        from langchain_core.messages import ToolMessage

        from priority_lens.models.canonical_event import EventType

        # Mock graph that returns tool calls and results
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Show my inbox"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_priority_inbox",
                                "args": {"limit": 10},
                                "id": "call_123",
                            }
                        ],
                    ),
                    ToolMessage(content="[]", tool_call_id="call_123"),
                    AIMessage(content="Your inbox is empty."),
                ]
            }
        )

        with patch("priority_lens.agent.graph.create_agent_graph", return_value=mock_graph):
            runner = AgentRunner(agent_context)
            events_gen = await runner.run_streaming("Show my inbox")

            # Collect events
            events = [event async for event in events_gen]

            # Should have: tool_call, tool_result, text_final
            assert len(events) == 3

            # First: tool call
            assert events[0].event_type == EventType.TOOL_CALL
            assert events[0].payload["tool_name"] == "get_priority_inbox"
            assert events[0].payload["tool_call_id"] == "call_123"

            # Second: tool result
            assert events[1].event_type == EventType.TOOL_RESULT
            assert events[1].payload["tool_call_id"] == "call_123"

            # Third: final text
            assert events[2].event_type == EventType.ASSISTANT_TEXT_FINAL
            assert events[2].payload["text"] == "Your inbox is empty."


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
