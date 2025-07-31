# CSV Data Cleaning & Reporting Agent (LangGraph)

[![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml)

This project demonstrates a robust, automated CSV data cleaning and reporting agent built with [LangGraph](https://github.com/langchain-ai/langgraph). The agent analyzes a CSV file, performs intelligent cleaning (handling nulls, outliers, and basic verification), and generates a comprehensive professional report summarizing trends, statistics, and the cleaning process.

## Features

- **Automated CSV analysis**: Inspects columns, types, missing values, and basic statistics.
- **Intelligent cleaning**: Handles missing values, outliers, and basic data verification.
- **No feature engineering or dimensionality reduction**: Focuses strictly on cleaning and validation.
- **Iterative refinement**: Cleans and verifies until data quality is satisfactory.
- **Comprehensive reporting**: Generates a professional report with trends, insights, statistics (mean, std, min, max), and cleaning summary.
- **All code execution is sandboxed**: Python code is executed in a subprocess for safety.
- **No plotting or visualization**: Only pandas and numpy are used.

## Directory Structure

```
graph_proj/
├── src/
│   └── agent/
│       ├── tool_call_agent.py   # Main agent logic
│       ├── dirty.csv            # Example input CSV (place your data here)
│       └── dirty_cleaned.csv    # Cleaned CSV output (auto-generated)
│       └── dirty_report.txt     # Professional report (auto-generated)
├── static/
│   └── studio_ui.png            # LangGraph Studio UI screenshot
├── .env.example                 # Example environment variables
├── README.md                    # This file
```

## Getting Started

### 1. Install dependencies

Install the project and LangGraph CLI:

```bash
cd /media/ronald/Kingston\ Shared/agent_data/graph_proj
pip install -e . "langgraph-cli[inmem]"
```

### 2. Prepare your environment

Copy the example environment file and set your API keys if needed:

```bash
cp .env.example .env
```

If you want to enable LangSmith tracing, add your LangSmith API key to `.env`:

```
LANGSMITH_API_KEY=lsv2...
```

### 3. Add your CSV data

Place your CSV file as `dirty.csv` in `src/agent/` (or update the code to use your filename).

### 4. Start the LangGraph Server

```bash
langgraph dev
```

This will launch the server and enable you to interact with the agent via LangGraph Studio.

## How It Works

- The agent loads `dirty.csv` and performs an initial analysis (columns, types, missing values, stats).
- It iteratively cleans the data (fills/drops nulls, handles outliers, verifies values).
- After each cleaning step, it saves the DataFrame as `dirty_cleaned.csv` (never overwriting the original).
- Once satisfied, it generates a comprehensive report (`dirty_report.txt`) summarizing:
  - Trends, patterns, and qualitative insights
  - Cleaning and verification steps performed
  - Answers to common questions (mean, std, min, max, etc.)
  - Remaining issues or recommendations

## Customization

- **Change the CSV filename**: Edit `init_csv_path` in `src/agent/tool_call_agent.py`.
- **Modify cleaning logic**: Update the system prompt in `summarizer_node` to adjust cleaning rules.
- **Extend reporting**: Enhance the `create_report` tool or prompt for more detailed reports.

## Development

- Edit code in `src/agent/tool_call_agent.py` to change agent behavior.
- Use LangGraph Studio for visual debugging and workflow editing.
- Hot reload is supported for rapid iteration.

## Example Output

After running, you will find:

- `dirty_cleaned.csv`: The cleaned version of your data.
- `dirty_report.txt`: A professional report, e.g.:

```
### Data Cleaning Report

- **Trends and Patterns**: 
  - The dataset contains personal details of individuals, with a focus on professional data like job titles and salaries. There is a high variation in salaries, indicating a diverse set of roles or locations.
- **Cleaning Summary**: 
  - Handled missing values by filling with median (Age, Salary) or mode (Job Title, Subscribed, Date Joined).
  - Capped extreme salary values at the 95th percentile to manage outliers.
  - Ensured consistent capitalization for country names and corrected job title typos.
- **Statistics**:
  - Mean, std, min, max, and other relevant statistics are included for all numeric columns.
- **Remaining Issues**:
  - Further validation of 'Date Joined' for future dates could be beneficial.
  - Consider future data collection improvements to reduce missing data.
```

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)
- [LangSmith](https://smith.langchain.com/)

---

*This project is a template for robust, automated CSV data cleaning and reporting using LangGraph and LLM-powered agents.*
