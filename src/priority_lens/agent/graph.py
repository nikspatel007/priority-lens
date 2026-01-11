"""LangGraph graph definition for Priority Lens agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal, cast

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from priority_lens.agent.state import AgentState
from priority_lens.agent.tools import PRIORITY_LENS_TOOLS, TOOL_EXECUTORS
from priority_lens.models.canonical_event import EventType
from priority_lens.services.agent_streaming import AgentEvent

if TYPE_CHECKING:
    from priority_lens.agent.context import AgentContext

logger = structlog.get_logger(__name__)


def create_agent_graph(
    model_name: str = "claude-sonnet-4-20250514",
) -> CompiledStateGraph[AgentState, AgentState]:
    """Create the Priority Lens agent graph.

    Args:
        model_name: Anthropic model to use.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # Create the model with tools bound
    model = ChatAnthropic(model_name=model_name, temperature=0)  # type: ignore[call-arg]
    model_with_tools = model.bind_tools(PRIORITY_LENS_TOOLS)

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Determine if agent should continue to tools or end.

        Args:
            state: Current agent state.

        Returns:
            "tools" if there are tool calls, "__end__" otherwise.
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Check if it's an AI message with tool calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "__end__"

    async def call_model(state: AgentState) -> dict[str, list[Any]]:
        """Invoke the model with the current state.

        Args:
            state: Current agent state.

        Returns:
            Updated state with new message.
        """
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph: StateGraph[AgentState] = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(PRIORITY_LENS_TOOLS))

    # Set entry point
    graph.set_entry_point("agent")

    # Add edges
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()  # type: ignore[return-value]


async def invoke_agent_with_context(
    graph: CompiledStateGraph[AgentState, AgentState],
    ctx: AgentContext,
    user_message: str,
) -> AgentState:
    """Invoke the agent with context for tool execution.

    This is the main entry point for running the agent with
    Priority Lens service access.

    Args:
        graph: Compiled agent graph.
        ctx: Agent context with user/session info.
        user_message: The user's input message.

    Returns:
        Final agent state with all messages.
    """
    from langchain_core.messages import HumanMessage

    initial_state: AgentState = {"messages": [HumanMessage(content=user_message)]}

    # For now, use the default tool node which doesn't have context
    # Full context integration will be added in Iteration 7
    result = await graph.ainvoke(initial_state)

    return cast(AgentState, result)


class AgentRunner:
    """Runner for the Priority Lens agent with context injection.

    This class provides the interface between the Turn Service
    and the LangGraph agent, handling context injection and
    event emission.
    """

    _ctx: AgentContext
    _graph: CompiledStateGraph[AgentState, AgentState]

    def __init__(self, ctx: AgentContext) -> None:
        """Initialize the agent runner.

        Args:
            ctx: Agent context with user/session info.
        """
        self._ctx = ctx
        self._graph = create_agent_graph()

    async def run(self, user_message: str) -> list[Any]:
        """Run the agent with a user message.

        Args:
            user_message: The user's input.

        Returns:
            List of messages from the agent run.
        """
        result = await invoke_agent_with_context(self._graph, self._ctx, user_message)
        return result["messages"]

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Any:
        """Execute a tool with the agent context.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments for the tool.

        Returns:
            Tool execution result.

        Raises:
            ValueError: If tool_name is not recognized.
        """
        executor = TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        await logger.ainfo(
            "tool_executing",
            tool=tool_name,
            args=tool_args,
            thread_id=str(self._ctx.thread_id),
        )

        # Call the executor with context and tool args
        # The executor functions have different signatures, so we use Any
        exec_func = cast(Any, executor)
        result = await exec_func(self._ctx, **tool_args)

        await logger.ainfo(
            "tool_executed",
            tool=tool_name,
            thread_id=str(self._ctx.thread_id),
        )

        return result

    async def run_streaming(
        self,
        user_message: str,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent and stream events.

        This method runs the LangGraph agent and yields AgentEvents
        for each significant step (text output, tool calls, tool results).

        Args:
            user_message: The user's input message.

        Yields:
            AgentEvent for each agent action.
        """
        from langchain_core.messages import HumanMessage

        async def generate() -> AsyncGenerator[AgentEvent, None]:
            initial_state: AgentState = {"messages": [HumanMessage(content=user_message)]}

            await logger.ainfo(
                "agent_streaming_start",
                thread_id=str(self._ctx.thread_id),
                user_message=user_message[:100],
            )

            # Use astream_events for real streaming when available
            # For now, invoke and convert final messages to events
            result = await self._graph.ainvoke(initial_state)
            messages = cast(AgentState, result)["messages"]

            # Convert messages to events
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    # Skip - already handled by TurnService
                    continue
                elif isinstance(msg, AIMessage):
                    # Check for tool calls
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            yield AgentEvent(
                                event_type=EventType.TOOL_CALL,
                                payload={
                                    "tool_name": tool_call["name"],
                                    "tool_args": tool_call["args"],
                                    "tool_call_id": tool_call.get("id", ""),
                                },
                            )
                    # Yield text content if present
                    if msg.content and isinstance(msg.content, str):
                        yield AgentEvent(
                            event_type=EventType.ASSISTANT_TEXT_FINAL,
                            payload={"text": msg.content},
                        )
                elif isinstance(msg, ToolMessage):
                    yield AgentEvent(
                        event_type=EventType.TOOL_RESULT,
                        payload={
                            "tool_call_id": msg.tool_call_id,
                            "result": str(msg.content)[:1000],  # Truncate large results
                        },
                    )

            await logger.ainfo(
                "agent_streaming_complete",
                thread_id=str(self._ctx.thread_id),
                message_count=len(messages),
            )

        return generate()
