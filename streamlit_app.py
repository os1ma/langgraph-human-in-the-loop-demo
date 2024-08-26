from typing import Literal, TypedDict
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph


@tool
def weather_search(city: str) -> str:
    """Search for the weather"""
    print("----")
    print(f"Searching for: {city}")
    print("----")
    return "Sunny!"


class State(MessagesState):
    """Simple state."""


def call_llm(state):
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools([weather_search])
    return {"messages": [model.invoke(state["messages"])]}


def human_review_node(state: dict) -> None:
    pass


def run_tool(state: dict) -> dict:
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


def route_after_llm(state: dict) -> Literal[END, "human_review_node"]:
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
        return "human_review_node"


def route_after_human(state: dict) -> Literal["run_tool", "call_llm"]:
    if isinstance(state["messages"][-1], AIMessage):
        return "run_tool"
    else:
        return "call_llm"


def create_graph() -> CompiledStateGraph:
    builder = StateGraph(State)
    builder.add_node(call_llm)
    builder.add_node(run_tool)
    builder.add_node(human_review_node)
    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", route_after_llm)
    builder.add_conditional_edges("human_review_node", route_after_human)
    builder.add_edge("run_tool", "call_llm")

    # Set up memory
    memory = MemorySaver()

    # Add
    return builder.compile(checkpointer=memory, interrupt_before=["human_review_node"])


def run_agent(
    graph: CompiledStateGraph,
    graph_input: dict | None,
    thread: dict,
) -> None:
    for event in graph.stream(graph_input, thread, stream_mode="values"):
        last_message = event["messages"][-1]

        if isinstance(last_message, AIMessage):
            if len(last_message.tool_calls) != 0:
                st.write("AgentがTool callを希望しています")
                st.write(last_message)
            else:
                st.write(last_message.content)


def app() -> None:
    load_dotenv(override=True)

    st.title("LangGraphでのHuman-in-the-loopの実装")

    # st.session_stateにthread_idを保存
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id

    graph = create_graph()

    # ex) what's the weather in sf?
    user_message = st.text_input("")

    if not user_message:
        return

    # Input
    initial_input = {"messages": [{"role": "user", "content": user_message}]}

    # Thread
    thread = {"configurable": {"thread_id": thread_id}}

    # Run the graph until the first interruption
    run_agent(graph, initial_input, thread)

    # 次のノードがhuman_review_nodeでない場合は終了
    next_node = graph.get_state(thread).next
    if len(next_node) == 0 or next_node[0] != "human_review_node":
        return

    approved = st.button("承認")

    if not approved:
        return

    run_agent(graph, None, thread)


app()
