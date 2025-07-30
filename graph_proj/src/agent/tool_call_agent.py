from typing import Annotated
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import OpenAI
from langgraph.prebuilt import ToolNode, tools_condition

from asyncio import tools

# Load environment variables from .env file
load_dotenv()

class State(TypedDict):
    thoughts : Annotated[list, add_messages]
    csv_path : str | None



graph_builder = StateGraph(State)


# Initialize the chat model without a system argument (handled manually)
llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.7)



def init_csv_path(state: State):
    if "csv_path" not in state:
        # the dirty.csv should be located in the same dir as this script
        state["csv_path"] = os.path.join(os.path.dirname(__file__), "dirty.csv")
        # lets test if the file exists and continue , otherwise, stop the graph
        if not os.path.exists(state["csv_path"]):
            raise FileNotFoundError(f"CSV file not found: {state['csv_path']}") 
    return state



def summarizer_node(state: State):
    if not isinstance(state["thoughts"][0], SystemMessage):
        system_message = SystemMessage(content="You are a summarizer of a given csv file. You will be given a csv file path and you will summarize the contents of the file. But you are not given to injest the whole csv file, you are only able to analyze it like a human would. An example is using pandas to read a few of them, not limiting to columns, total rows etc.. You cannot run the pandas code yourself, however, you are available to call a tool that can do this for you. You need to just give pure working python code that uses pandas, in the python code, you are able to see the output of certain variables by using print as the tool will capture all std output and throw it back to you later, but use print sparingly and not print very long outputs. ")
        
        return {"thoughts": [system_message]}

def checkpoint(state: State):
    # This function is a placeholder for any checkpoint logic you might want to implement
    # For now, it just returns the state unchanged
    return state

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
tools = []

tool_node = ToolNode(tools=tools)

graph_builder.add_node("init_csv_path", init_csv_path)
graph_builder.add_node("checkpoint", checkpoint)
graph_builder.add_edge(START, "init_csv_path")
graph_builder.add_edge("init_csv_path", "checkpoint")
graph_builder.add_edge("checkpoint", END)
graph = graph_builder.compile()