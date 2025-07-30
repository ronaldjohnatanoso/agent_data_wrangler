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
    analysis_trace: list  # just a plain list now, updated by nodes

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
    analysis_trace = state.get("analysis_trace", [])

    # If there's an execution_result to append, add it as a structured message in history,
    # (Only update messages and feedback here, NOT analysis_trace. The tool node already records stdout.)
    if execution_result is not None:
        # Provide very explicit output & error separation as an AI message for the LLM
        if execution_result.get("success"):
            feedback = (
                f"Execution succeeded.\nSTDOUT:\n{execution_result.get('stdout','')}\n"
            )
        else:
            feedback = (
                f"Execution failed.\nSTDOUT:\n{execution_result.get('stdout','')}\nSTDERR:\n{execution_result.get('stderr','')}\n"
                "You MUST analyze the error, correct your previous code, and only call execute_python_code with the FIXED version. Never repeat bad code."
            )
        exec_message = AIMessage(content=feedback)
        messages = messages + [exec_message]

    # If previous execution_result was successful, summarize the result and stop (no PDF)
    if execution_result is not None and execution_result.get("success"):
        # Optionally ask the LLM to summarize findings in text, or just append as is
        stats_text = execution_result.get("stdout", "")
        findings_prompt = (
            "Given the following CSV analysis output, write a concise (no more than 3-5 sentences), professional summary (plain text, no PDF, no markup). "
            "Only state the most important findings, missing values, and data characteristics. Be brief, skip unnecessary filler, and avoid lengthy explanations.\n\n"
            f"{stats_text}"
        )
        findings = llm.invoke([HumanMessage(content=findings_prompt)]).content
        analysis_trace = analysis_trace + [{"type": "analysis", "content": findings}]
        final_message = AIMessage(content=findings)
        return {
            "messages": messages + [final_message],
            "analysis_trace": analysis_trace
        }

    # If we reach here, either no execution yet or last execution had error; prompt for code/tool-call.
    system_message = SystemMessage(
        content=(
            f"You are a CSV analyst subprocess. "
            f"Your ONLY permitted libraries are pandas and numpy (no plotting, visualization, seaborn, matplotlib, etc). "
            f"Every code you generate must be a FULL, standalone script: always import pandas as pd (and numpy if needed), and load the CSV file from '{csv_path}' into a DataFrame named df at the start."
            " Do NOT assume any variables exist unless you explicitly create them in each output."
            "\nYour task is ONLY to explore the basic shape of the CSV: print the columns, print the number of rows and columns, print df.head(5), show df.dtypes, print basic statistics with df.describe(), and print counts of missing/null values per column. Do not do feature engineering, wrangling, or advanced analysis. "
            "Strictly avoid any plotting or data visualization commands."
            f"\nCSV path: {csv_path}"
            "\n\nIMPORTANT:"
            "\n- For all print statements, always use a SINGLE, one-line print statement per value. NEVER use '\\n' inside string literals, and NEVER use multi-line strings or triple quotes. For each statistic, use code like: print('Columns:', df.columns.tolist()), print('Number of Rows:', df.shape[0]) and so on."
            "\n- Do not join labels and output with newlines. For example, instead of print('Basic statistics:\\n', df.describe()), always use print('Basic statistics:', df.describe()) or similar."
            "\n- DO NOT use triple-quoted strings or f-strings that span multiple lines inside print. Print each output and its label as a single expression."
            "\n- If code execution fails or returns an error message, carefully review any error output and fix your code before retrying (never repeat unrevised code, always correct mistakes or syntax as indicated by the error message)."
            "\n- ONLY call the execute_python_code tool when producing code for execution. Do NOT output a code block or direct text to the user, and do NOT repeat previous code if it has already failed."
            "\n- If there is an execution error, always explain the root cause to yourself before writing the corrected code."
        )
    )
    human_message = HumanMessage(content="Analyze the CSV file and, if needed, write code for further analysis. If required, call the execute_python_code tool.")

    response = llm_with_tools.invoke([system_message] + messages + [human_message])
    return {
        "messages": messages + [response],
        "analysis_trace": analysis_trace
    }
    

def should_continue(state: State):
    """Determine if we should continue to tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    execution_result = state.get("execution_result", None)
    # Allow tool calls if present (even if prev execution was successful, since that may be for PDF next)
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

def tool_node_with_analysis_trace(state: State):
    """
    Executes the tool (assumes only 'execute_python_code' will be called) and updates analysis_trace in the state.
    """
    messages = state.get("messages", [])
    analysis_trace = state.get("analysis_trace", []) if "analysis_trace" in state else []
    execution_result = None

    # Find the last message and its tool call
    last_message = messages[-1] if messages else None
    tool_call = None
    if last_message and hasattr(last_message, 'tool_calls'):
        for call in last_message.tool_calls:
            if call.get("name") == "execute_python_code":
                tool_call = call
                break

    # Run the tool and update state, including analysis_trace
    if tool_call:
        args = tool_call.get("args", {})
        code = args.get("code")
        execution_result = execute_python_code(code)
        if execution_result and execution_result.get("stdout"):
            analysis_trace = analysis_trace + [{"type": "stdout", "content": execution_result.get("stdout", "")}]
    else:
        execution_result = None  # Defensive

    return {
        **state,
        "execution_result": execution_result,
        "analysis_trace": analysis_trace,
    }

tool_node = tool_node_with_analysis_trace

graph_builder.add_node("init_csv_path", init_csv_path)  
graph_builder.add_node("summarizer", summarizer_node)
graph_builder.add_node("tools", tool_node)
# After one tool execution, always halt (END).

graph_builder.add_edge(START, "init_csv_path")  
graph_builder.add_edge("init_csv_path", "summarizer")  
graph_builder.add_conditional_edges("summarizer", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "summarizer")

graph = graph_builder.compile()