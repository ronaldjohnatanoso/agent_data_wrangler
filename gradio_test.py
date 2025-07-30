import gradio as gr

# Maximum number of blocks that can be added in the interface
MAX_BLOCKS = 100

def add_block(*args):
    """
    Add a new block by increasing the visibility of the next available textbox.
    Args:
        *args: Values of the current blocks followed by the current block count.
    Returns:
        A list of updates for the visibility and value of textboxes, and the new block count.
    """
    # Extract the current values of the textboxes
    values = list(args[:-1])
    count = args[-1]  # The current count of active blocks

    # Increment the block count if below the maximum limit
    if count < MAX_BLOCKS:
        count += 1

    # Update the visibility and preserve existing values for textboxes
    return [gr.update(visible=i < count, value=values[i]) for i in range(MAX_BLOCKS)] + [count]

def clear_blocks():
    """
    Clear all blocks by hiding all textboxes and resetting their values.
    Returns:
        A list of updates setting all textboxes to be hidden and empty, and resetting the block count to 0.
    """
    return [gr.update(visible=False, value="") for _ in range(MAX_BLOCKS)] + [0]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧱 Editable Dynamic Blocks")  # Title of the app

    # State to track the current number of visible blocks
    block_count = gr.State(0)

    # Buttons for adding and clearing blocks
    add_btn = gr.Button("➕ Add Block")  # Button to add new blocks
    clear_btn = gr.Button("🗑️ Clear All")  # Button to clear all blocks

    # Create 10 initially hidden textboxes to represent the blocks
    inputs = [gr.Textbox(label=f"Block #{i+1}", visible=False) for i in range(MAX_BLOCKS)]

    # Connect the add button to the add_block function
    add_btn.click(
        fn=add_block,
        inputs=inputs + [block_count],  # Inputs include the current textbox values and block count
        outputs=inputs + [block_count]  # Outputs include updated textbox states and block count
    )

    # Connect the clear button to the clear_blocks function
    clear_btn.click(
        fn=clear_blocks,
        outputs=inputs + [block_count]  # Outputs hide all textboxes and reset the block count
    )

# Launch the application
demo.launch()
