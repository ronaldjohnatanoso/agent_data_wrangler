from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

def format_message(data):
    for message in data.get('messages', []):
        role = message.get('type', 'unknown').capitalize()
        content = message.get('content', '[No content]')
        if role == 'Human':
            print(f"human: {content}")
        elif role == 'Ai':
            print(f"ai: {content}")
        else:
            print(f"other ({role}): {content}")

for chunk in client.runs.stream(
    None,  # Thread ID set to "1"
    "agent", # Name of assistant. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",
            "content": "Who killed magellan?",
        }],
    },
    stream_mode="values",
):
    format_message(chunk.data)
    print("\n")