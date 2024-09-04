from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel.types import StateSnapshot


@tool
def weather_search(city: str) -> str:
    """Search for the weather"""
    return "Sunny!"


class HumanInTheLoopAgentState(MessagesState):
    """Simple state."""


class HumanInTheLoopAgent:
    def __init__(self) -> None:
        builder = StateGraph(HumanInTheLoopAgentState)
        builder.add_node("call_llm", self._call_llm)
        builder.add_node("run_tool", self._run_tool)
        builder.add_node("human_review_node", self._human_review_node)
        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", self._route_after_llm)
        builder.add_conditional_edges("human_review_node", self._route_after_human)
        builder.add_edge("run_tool", "call_llm")

        memory = MemorySaver()

        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["human_review_node"],
        )

    def _call_llm(self, state: dict) -> dict:
        model = ChatOpenAI(model="gpt-4o-mini").bind_tools([weather_search])
        return {"messages": [model.invoke(state["messages"])]}

    def _human_review_node(self, state: dict) -> None:
        pass

    def _run_tool(self, state: dict) -> dict:
        new_messages = []
        tools = {"weather_search": weather_search}
        tool_calls = state["messages"][-1].tool_calls
        for tool_call in tool_calls:
            tool = tools[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            new_messages.append(
                {
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": result,
                    "tool_call_id": tool_call["id"],
                }
            )
        return {"messages": new_messages}

    def _route_after_llm(self, state: dict) -> Literal[END, "human_review_node"]:
        if len(state["messages"][-1].tool_calls) == 0:
            return END
        else:
            return "human_review_node"

    def _route_after_human(self, state: dict) -> Literal["run_tool", "call_llm"]:
        if isinstance(state["messages"][-1], AIMessage):
            return "run_tool"
        else:
            return "call_llm"

    def run(self, graph_input: dict | None, thread_id: str) -> None:
        for _ in self.graph.stream(
            input=graph_input,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        ):
            pass

    def get_messages(self, thread_id: str) -> Any:
        return self._get_state(thread_id).values["messages"]  # noqa: PD011

    def is_next_human_review_node(self, thread_id: str) -> bool:
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_review_node"

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config={"configurable": {"thread_id": thread_id}})

    def mermaid_png(self) -> bytes:
        return self.graph.get_graph().draw_mermaid_png()
