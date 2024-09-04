from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import HumanInTheLoopAgent


def show_messages(messages: list[Any]) -> None:
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message(message.type):
                st.write(message.content)

        elif isinstance(message, AIMessage):
            # tool_callの場合はツールの承認を求める旨を表示
            if len(message.tool_calls) != 0:
                for tool_call in message.tool_calls:
                    with st.chat_message(message.type):
                        st.write("エージェントがツールの承認を求めています")
                        st.write(f"ツール名: {tool_call['name']}")
                        st.write(f"引数: {tool_call['args']}")
            else:
                with st.chat_message(message.type):
                    st.write(message.content)

        elif isinstance(message, ToolMessage):
            with st.chat_message(message.type):
                st.write("ツールの実行結果")
                st.write(message.content)

        else:
            raise ValueError(f"Unknown message type: {type(message)}")


def app() -> None:
    load_dotenv(override=True)

    st.title("LangGraphでのHuman-in-the-loopの実装")

    # st.session_stateにagentを保存
    if "agent" not in st.session_state:
        st.session_state.agent = HumanInTheLoopAgent()
    agent = st.session_state.agent

    # グラフを表示
    with st.sidebar:
        st.image(agent.mermaid_png())

    # st.session_stateにthread_idを保存
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    st.write(f"thread_id: {thread_id}")

    # ユーザーの指示を受け付ける
    user_message = st.chat_input()
    if user_message:
        graph_input = {"messages": HumanMessage(content=user_message)}
        with st.spinner():
            agent.run(graph_input, thread_id)

    # 会話履歴を表示
    messages = agent.get_messages(thread_id)
    show_messages(messages)

    # 次がhuman_review_nodeの場合は承認ボタンを表示
    if agent.is_next_human_review_node(thread_id):
        approved = st.button("承認")
        # 承認されたらエージェントを実行
        if approved:
            with st.spinner():
                agent.run(None, thread_id)
            # 会話履歴を表示するためrerun
            st.rerun()


app()
