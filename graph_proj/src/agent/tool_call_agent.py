from typing import Annotated  
import os  
import tempfile  
import subprocess  
from dotenv import load_dotenv  
  
from langchain.chat_models import init_chat_model  
from langchain_core.tools import tool
from typing_extensions import TypedDict  
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  
  
from langgraph.graph import StateGraph, START, END  
from langgraph.graph.message import add_messages  
from langgraph.prebuilt import ToolNode, tools_condition  

# Load environment variables from .env file
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    csv_path: str | None
    code_to_execute: str | None
    execution_result: dict | None
    summary: str | None

# Initialize the chat model
llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.7)

@tool
def execute_python_code(code: str) -> dict:
    """Executes pure Python code and captures stdout and stderr."""  

    try:  
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_code_file:  
            temp_code_file.write(code.encode())  
            temp_code_file_path = temp_code_file.name  

        # Execute the code in a subprocess to capture output and errors  
        process = subprocess.run(  
            ["python3", temp_code_file_path], capture_output=True, text=True  
        )  
        stdout = process.stdout.strip()  
        stderr = process.stderr.strip()  

        # Cleanup the temporary file  
        os.unlink(temp_code_file_path)  

        if process.returncode != 0:  
            return {"success": False, "stdout": stdout, "stderr": stderr}  

        return {"success": True, "stdout": stdout, "stderr": stderr}  

    except Exception as e:  
        return {"success": False, "stdout": "", "stderr": str(e)}

# Bind tools to the LLM
tools = [execute_python_code]
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)

def init_csv_path(state: State):
    if state.get("csv_path") is None:
        # the dirty.csv should be located in the same dir as this script
        csv_path = os.path.join(os.path.dirname(__file__), "dirty.csv")
        # lets test if the file exists and continue, otherwise, stop the graph
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}") 
        state["csv_path"] = csv_path
    return state

def summarizer_node(state: State):
    """
    Main director node for CSV analysis.
    Uses only message (and appended execution result) history as context for reasoning.
    No 'store_basic_info', just iterative code analysis/refinement.
    """
    csv_path = state.get("csv_path", "")
    messages = state.get("messages", [])
    execution_result = state.get("execution_result", None)

    # If there's an execution_result to append, add it as a message in history
    # We wrap it as an AIMessage for LLM visibility; user code can customize the structure if wanted
    if execution_result is not None:
        exec_message = AIMessage(content=str(execution_result))
        messages = messages + [exec_message]

    # Summarizer always analyzes whole conversation + execution results so far
    system_message = SystemMessage(
        content=(
            f"You are a CSV analyst operating interactively. "
            f"Your job is to reason through the history provided. "
            f"Write Python (pandas) code to: "
            "- Identify and summarize columns, row/column counts, data types, nulls, and anomalies; "
            "- Sample 10 rows, high-level nulls, and stats; "
            "- When these are known, proceed to more advanced/exploratory analysis (outliers, trends, etc). "
            "At every turn, print findings and only generate the code actually needed to progress further. "
            f"The CSV path is: {csv_path}\n"
            "IMPORTANT: Only use variables that you define in the code you generate. Do NOT print variables such as 'summary' unless you have actually created them in your code. Make all code self-contained."
            "\nOutput code for the next needed step."
        )
    )
    human_message = HumanMessage(content="Analyze the CSV file and, if needed, write code for further analysis.")

    response = llm_with_tools.invoke([system_message] + messages + [human_message])
    return {
        "messages": messages + [response]
    }
    

def should_continue(state: State):
    """Determine if we should continue to tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return END
        
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

graph_builder.add_node("init_csv_path", init_csv_path)  
graph_builder.add_node("summarizer", summarizer_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "init_csv_path")  
graph_builder.add_edge("init_csv_path", "summarizer")  
graph_builder.add_conditional_edges("summarizer", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "summarizer")

graph = graph_builder.compile()