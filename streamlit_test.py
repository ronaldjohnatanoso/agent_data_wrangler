import streamlit as st

st.set_page_config(page_title="Dynamic Block Creator", layout="centered")
st.title("🚀 Dynamic Block Creator")

# Init session state
if "blocks" not in st.session_state:
    st.session_state.blocks = []

# UI: Add & Clear Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Block"):
        idx = len(st.session_state.blocks)
        st.session_state.blocks.append({
            "label": f"Block {idx + 1}",
            "key": f"block_{idx}",
            "value": ""
        })
        st.session_state._rerun_trigger = True  # Force rerun
        st.experimental_set_query_params(rerun="1")  # workaround

with col2:
    if st.button("🗑️ Clear All"):
        for block in st.session_state.blocks:
            if block["key"] in st.session_state:
                del st.session_state[block["key"]]
        st.session_state.blocks = []
        st.session_state._rerun_trigger = True
        st.experimental_set_query_params(rerun="1")

st.markdown("---")

# UI: Modify Block
target_label = st.text_input("Enter block label to modify (e.g., Block 1):")
new_text = st.text_input("New text to set for that block:")

if st.button("Modify Block"):
    found = False
    for block in st.session_state.blocks:
        if block["label"] == target_label:
            block["value"] = new_text
            found = True
            st.success(f"✅ {target_label} updated.")
            st.session_state._rerun_trigger = True
            st.experimental_set_query_params(rerun="1")
            st.stop()  # clean exit before render
    if not found:
        st.error(f"❌ Block '{target_label}' not found.")

st.markdown("---")

# Render blocks
for block in st.session_state.blocks:
    block["value"] = st.text_input(
        label=block["label"],
        value=block["value"],
        key=block["key"]
    )
