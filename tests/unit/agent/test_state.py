"""Tests for agent state."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from priority_lens.agent.state import AgentState


class TestAgentState:
    """Tests for AgentState."""

    def test_state_creation(self) -> None:
        """Test creating an agent state."""
        state: AgentState = {"messages": []}
        assert state["messages"] == []

    def test_state_with_messages(self) -> None:
        """Test creating state with messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        state: AgentState = {"messages": messages}
        assert len(state["messages"]) == 2
        assert state["messages"][0].content == "Hello"
        assert state["messages"][1].content == "Hi there!"

    def test_state_message_types(self) -> None:
        """Test that state accepts different message types."""
        human = HumanMessage(content="Test question")
        ai = AIMessage(content="Test answer")

        state: AgentState = {"messages": [human, ai]}

        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
