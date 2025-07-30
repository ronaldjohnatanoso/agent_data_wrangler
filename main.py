import gradio as gr

# Global list to store block components
block_components = []

def add_block():
    """Add a new block component"""
    block_num = len(block_components) + 1
    new_block = gr.Textbox(
        label=f"Block #{block_num}",
        placeholder="Your text here...",
        lines=2,
        visible=True
    )
    block_components.append(new_block)
    
    # Return visibility updates for all blocks
    return [gr.update(visible=True) for _ in block_components] + [gr.update(visible=False) for _ in range(10 - len(block_components))]

def clear_blocks():
    """Clear all blocks"""
    global block_components
    block_components = []
    # Hide all blocks
    return [gr.update(visible=False) for _ in range(10)]

# Create interface with pre-created hidden blocks
with gr.Blocks() as demo:
    gr.Markdown("# Simple Block Generator")
    
    add_btn = gr.Button("Add Block")
    clear_btn = gr.Button("Clear All")
    
    # Pre-create 10 hidden blocks
    blocks = []
    for i in range(10):
        block = gr.Textbox(
            label=f"Block #{i+1}",
            placeholder="Your text here...",
            lines=2,
            visible=False
        )
        blocks.append(block)
    
    # Button handlers
    add_btn.click(
        fn=add_block,
        outputs=blocks
    )
    
    clear_btn.click(
        fn=clear_blocks,
        outputs=blocks
    )

if __name__ == "__main__":
    demo.launch()