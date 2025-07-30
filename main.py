import gradio as gr

# Global list to store blocks
blocks = []

def add_block():
    """Add a new simple block"""
    block_num = len(blocks) + 1
    blocks.append(f"Block #{block_num}: Your text here...")
    return blocks

def clear_blocks():
    """Clear all blocks"""
    global blocks
    blocks = []
    return blocks

# Create simple interface
with gr.Blocks() as demo:
    gr.Markdown("# Simple Block Generator")
    
    add_btn = gr.Button("Add Block")
    clear_btn = gr.Button("Clear All")
    
    # Display blocks as a simple list
    block_display = gr.Textbox(
        label="Blocks",
        lines=10,
        interactive=False,
        value=""
    )
    
    # Button handlers
    add_btn.click(
        fn=lambda: "\n\n".join(add_block()),
        outputs=block_display
    )
    
    clear_btn.click(
        fn=lambda: "\n\n".join(clear_blocks()),
        outputs=block_display
    )

if __name__ == "__main__":
    demo.launch()