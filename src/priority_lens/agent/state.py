"""Agent state definition for LangGraph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the Priority Lens agent.

    Attributes:
        messages: Conversation history with tool calls and results.
    """

    messages: Annotated[list[AnyMessage], add_messages]
