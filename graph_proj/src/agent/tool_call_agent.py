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
    basic_info: dict | None

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
        state["basic_info"] = None  # Initialize basic_info as None
    return state

def summarizer_node(state: State):
    """
    Main director node for CSV analysis. 
    1. If execution_result has action 'store_basic_info': store its data to basic_info.
    2. If basic_info missing: prompt LLM for basic info code (must output action {"action": "store_basic_info"} with info).
    3. If basic_info exists: prompt LLM for deeper analysis using pandas (nulls, outliers, etc.), with context.
    """
    import json

    csv_path = state.get("csv_path", "")
    basic_info = state.get("basic_info", None)
    execution_result = state.get("execution_result", None)
    messages = state.get("messages", [])

    # 1. If tool just ran and possibly gave us new basic_info, detect and store it
    if execution_result and isinstance(execution_result, dict):
        stdout = execution_result.get("stdout", "")
        captured = None
        try:
            # Try parsing stdout as a dict
            if isinstance(stdout, str) and "action" in stdout and "store_basic_info" in stdout:
                captured = json.loads(stdout)
            elif isinstance(stdout, dict):  # rare case: direct dict
                captured = stdout
        except Exception:
            pass
        if captured and captured.get("action") == "store_basic_info":
            return {
                "messages": messages,
                "basic_info": captured
            }

    # 2. If basic_info does NOT exist, prompt for basic info code with action signal at end
    if not basic_info:
        system_message = SystemMessage(
            content=(
                f"You are a CSV analyzer. To begin, write pandas Python code that extracts:"
                "\n  - The column names\n  - Total row/column counts\n  - Data types per column"
                "\n  - A sample of 10 rows\n  - Any basic null/missing/obvious anomaly stats"
                "\nAt the end, print as a single JSON dict like this:"
                '\n  {"action": "store_basic_info", "columns":[...],"dtypes":{...},"shape":[...],"sample":[...], ...}'
                "\nOnly print this JSON dict to stdout!"
                f"\nThe CSV path is: {csv_path}"
            )
        )
        human_message = HumanMessage(content="Extract CSV basic info and emit the JSON dict as described.")
        response = llm_with_tools.invoke([system_message, human_message])
        return {
            "messages": messages + [response],
            "basic_info": None
        }

    # 3. If basic_info exists, move to more advanced analysis
    system_message = SystemMessage(
        content=(
            f"You are a CSV data analyst. Using the existing basic info:"
            f"\n{basic_info}"
            "\nNow, analyze the data further in Python using pandas. Write code to:"
            "\n- Detect/quantify missing values"
            "\n- Identify outliers/numerical anomalies"
            "\n- Highlight notable statistics or trends"
            "\nPresent your findings with print statements, but do not re-extract basic info."
            f"\nThe CSV file location: {csv_path}."
        )
    )
    human_message = HumanMessage(content="Do a deeper exploratory analysis (nulls, outliers, weird stats, trends, etc).")
    response = llm_with_tools.invoke([system_message, human_message])
    return {
        "messages": messages + [response],
        "basic_info": basic_info
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