from typing import Annotated
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import OpenAI
import logging

# Load environment variables from .env file
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]



graph_builder = StateGraph(State)


# Initialize the chat model without a system argument (handled manually)
llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.7)


def chatbot_me(state: State):
    if not isinstance(state["messages"][0], SystemMessage):
        system_message = SystemMessage(content="You are a helpful assistant and address the user as 'your grace'")
        state["messages"].insert(0, system_message)

    response = llm.invoke(state["messages"])


    # print the whole messages list along with the response from ai then return the state
    print("Messages:", state["messages"])
    print("AI Response:", response)
    return {"messages": [response]}



# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot_me", chatbot_me)
graph_builder.add_edge(START, "chatbot_me")
graph_builder.add_edge("chatbot_me", END)
graph = graph_builder.compile()