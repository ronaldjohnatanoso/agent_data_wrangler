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
llm = init_chat_model(model="openai:gpt-4o", temperature=0.7)

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

@tool
def create_report(report_content: str, csv_path: str) -> dict:
    """
    Saves the provided report_content to a .txt file in the same directory as the csv_path.
    The report file will be named like the CSV but with '_report.txt' appended.
    """
    try:
        base = os.path.splitext(csv_path)[0]
        report_path = f"{base}_report.txt"
        with open(report_path, "w") as f:
            f.write(report_content)
        return {"success": True, "report_path": report_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Bind tools to the LLM
tools = [execute_python_code, create_report]
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
    Main director node for CSV analysis and cleaning.
    Uses message and execution result history as context for reasoning.
    At the end, produces a comprehensive professional report based on the analysis and cleaning performed.
    """
    csv_path = state.get("csv_path", "")
    messages = state.get("messages", [])
    execution_result = state.get("execution_result", None)

    # If there's an execution_result to append, add it as a structured message in history
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

    # System prompt: restrict to cleaning only, and require a comprehensive report at the end
    system_message = SystemMessage(
        content=(
            f"You are a CSV analyst subprocess. "
            f"Your ONLY permitted libraries are pandas and numpy (no plotting, visualization, seaborn, matplotlib, etc). "
            f"Every code you generate must be a FULL, standalone script: always import pandas as pd (and numpy if needed), and load the CSV file from '{csv_path}' into a DataFrame named df at the start."
            " Do NOT assume any variables exist unless you explicitly create them in each output."
            "\n\nEach Python code you generate must be a fully standalone script, with all necessary imports and CSV loading included."
            "\n\nYour task is to:"
            "\n- Explore the basic shape of the CSV: print the columns, print the number of rows and columns, print df.head(5), show df.dtypes, print basic statistics with df.describe(), and print counts of missing/null values per column."
            "\n- Intelligently decide if the data needs cleaning: for example, if there are missing values, decide whether to drop rows, fill with mean/median/mode, or otherwise impute. If there are outliers, consider removing or capping them. Justify your cleaning and verification choices in your reasoning."
            "\n- Perform all necessary cleaning, verification, and outlier handling until you are satisfied with the data quality."
            "\n- When you are satisfied, save the final cleaned DataFrame to a new CSV file in the same directory as the original, with '_cleaned' appended before the '.csv' extension (e.g., 'dirty_cleaned.csv'). Never overwrite the original file."
            "\n- After each mutation, always reload the DataFrame from the latest clean CSV for further analysis or cleaning."
            "\n- All print statements must output only a single line (no multi-line prints)."
            "\n- Do NOT repeat code or outputs that have already been executed or printed. Keep track of what you have already analyzed or displayed, and only perform new actions or move on to the next step."
            "\n- Do NOT repeat the initial exploration (columns, shape, head, dtypes, describe, null counts) if it has already been done; proceed to cleaning or further analysis."
            "\n- Do NOT repeat code that has already failed or been executed. Only retry if you have revised the code based on the error."
            "\n- If a task is completed or cannot be fixed after a reasonable attempt, move on to the next logical step or finish."
            "\n\nWhen you are fully satisfied with the cleaning and verification, produce a comprehensive professional report that includes:"
            "\n- Trends, patterns, and qualitative insights you noticed in the data (based only on what you have seen in outputs)."
            "\n- A detailed summary of the cleaning steps, verification, and outlier handling you performed."
            "\n- Answers to common questions about the data, such as mean, standard deviation, min, max, and any other relevant statistics."
            "\n- Any remaining issues or recommendations for further analysis."
            "\nCall the create_report tool with your report content and the CSV path to save the report as a .txt file in the same directory as the CSV."
            "\n\nIMPORTANT:"
            "\n- If code execution fails or returns an error message (from either stdout or stderr), carefully review any error output and fix your code before retrying (never repeat unrevised code, always correct mistakes or syntax as indicated by the error message)."
            "\n- ONLY call the execute_python_code tool when producing code for execution. Do NOT output a code block or direct text to the user, and do NOT repeat previous code if it has already failed."
            "\n- If there is an execution error, always explain the root cause to yourself before writing the corrected code."
            "\n- Never use plotting or visualization commands."
            f"\nCSV path: {csv_path}"
        )
    )
    human_message = HumanMessage(
        content=(
            "Analyze the CSV file, perform only cleaning (no feature engineering or dimensionality reduction), and if so, write code to perform it. "
            "If you mutate the DataFrame, always save to a new '_cleaned' CSV and reload from it. "
            "When you are satisfied, produce a comprehensive professional report with trends, insights, cleaning summary, statistics, and call the create_report tool to save it."
        )
    )

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