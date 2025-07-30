import streamlit as st

# Set up Streamlit page configuration
st.set_page_config(page_title="Dynamic Block Creator", layout="centered")
st.title("🚀 Dynamic Block Creator")

# Initialize the state to store blocks if not already present
if "blocks" not in st.session_state:
    st.session_state.blocks = []  # List to hold data for dynamically created blocks

# Create two buttons side by side: Add Block and Clear All
col1, col2 = st.columns(2)

with col1:
    # "➕ Add Block" button appends a new block with a unique label to the session state
    if st.button("➕ Add Block"):
        st.session_state.blocks.append({"label": f"Block {len(st.session_state.blocks) + 1}"})

with col2:
    # "🗑️ Clear All" button clears all blocks from the session state and resets the interface
    if st.button("🗑️ Clear All"):
        st.session_state.blocks = []  # Clear all blocks
        # st.experimental_rerun()  # (Optional) Can be used for immediate UI refresh

# Add a horizontal divider for separation
st.markdown("---")

# Define a function to handle input changes and print the block label
def on_input_change(label):
    # This function gets called when a user presses Enter in a text input
    st.write(f"Block identifier: {label}")

# Loop through each block in the session state and dynamically render the UI
for block in st.session_state.blocks:
    with st.container():
        # Display a text input for each block using its label as the key
        st.text_input(
            block["label"], 
            key=block["label"], 
            on_change=on_input_change, 
            args=(block["label"],)  # Pass the block's label to the callback function
        )
