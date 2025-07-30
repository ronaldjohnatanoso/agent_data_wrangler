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
    csv_path = state.get("csv_path", "")
    basic_info = state.get("basic_info", None)
    current_messages = state.get("messages", [])

    # If basic_info already exists, let the LLM know to avoid re-requesting
    if basic_info is not None:
        system_message = SystemMessage(
            content=f"""You are a CSV file analyzer. You will analyze the CSV file at: {csv_path}

            Basic info has already been extracted and stored. Use or refer to this info when answering follow-up requests. Don't repeat code that fetches basic information again unless specifically asked for it.

            BASIC INFO STATE:
            {basic_info}

            Continue handling the analysis from here based on user queries or the previous context."""
        )
    else:
        system_message = SystemMessage(
            content=f"""You are a CSV file analyzer. You will analyze the CSV file at: {csv_path}
            
            Your task is to:
            1. Write Python code using pandas to explore the CSV file
            2. Analyze columns, data types, sample data, and basic statistics
            3. Use print() statements sparingly to show key insights
            4. The code will be executed by a tool, so make it clean and functional
            
            Start by reading the CSV file and showing basic information about it."""
        )
    
    # Initialize messages if empty or add to existing
    if not current_messages:
        human_message = HumanMessage(content=f"Please analyze the CSV file at {csv_path} and provide Python code to explore its contents.")
        messages = [system_message, human_message]
    else:
        messages = current_messages

    response = llm_with_tools.invoke(messages)
    updated_messages = current_messages + [response]

    # Store basic_info only after getting a successful execution_result and it's not set yet
    new_basic_info = None

    if basic_info is None:
        exec_result = state.get("execution_result")
        if exec_result and isinstance(exec_result, dict) and exec_result.get("success"):
            # Store stdout or fallback to stderr as the basic info
            new_basic_info = exec_result.get("stdout", "") or exec_result.get("stderr", "")

    next_basic_info = basic_info or new_basic_info

    return {
        "messages": updated_messages,
        "basic_info": next_basic_info if next_basic_info else None
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