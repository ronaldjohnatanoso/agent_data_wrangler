FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (if you have requirements.txt)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Or, if you use setup.py/pyproject.toml, copy and install your package
# COPY . .
# RUN pip install -e . "langgraph-cli[inmem]"

# Copy the rest of the code
COPY . .

# Expose port for LangGraph server (default 8080)
EXPOSE 8080

# Default command: start LangGraph server
CMD ["langgraph", "dev"]
