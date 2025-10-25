import os
import json
import time
import streamlit as st
from openai import OpenAI
from typing import Generator, List, Dict

# ----------------------
# App config & helpers
# ----------------------
st.set_page_config(page_title="💬 OpenAI Chatbot", page_icon="💬", layout="centered")

st.title("💬 Chatbot (OpenAI API, streaming)")
st.caption(
    "A minimal but production-friendly Streamlit chat app using OpenAI's **Responses API** with streaming,"
    " model selector, temperature control, error handling, and chat persistence."
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    # Prefer secrets/env var over text input to avoid exposing the key in reruns/history.
    default_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    api_key = st.text_input("OpenAI API Key", type="password", value=default_api_key)
    model = st.selectbox(
        "Model",
        # Keep options concise; users can type a custom name as well.
        [
            "gpt-5",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4o-mini",
        ],
        index=2,
        help="Choose a chat-capable model. See OpenAI docs for the latest list."
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                            help="Higher values = more creative, lower = more deterministic")
    max_output_tokens = st.slider("Max output tokens", 128, 8192, 1024, 64)

    st.divider()
    system_prompt = st.text_area(
        "System prompt (optional)",
        value="You are a helpful, concise assistant.",
        help="Prepends a system message to steer the assistant's behavior.",
        height=100,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        clear_btn = st.button("🧹 Clear chat", use_container_width=True)
    with col_b:
        export_btn = st.button("💾 Export chat", use_container_width=True)

# ----------------------
# Session state
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if clear_btn:
    st.session_state.messages = []
    st.rerun()

if export_btn:
    fn = f"chat_{int(time.time())}.json"
    st.download_button(
        label="Download chat JSON",
        data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=fn,
        mime="application/json",
        use_container_width=True,
    )

# ----------------------
# Guardrail: API key
# ----------------------
if not api_key:
    st.info("Add your OpenAI API key in the sidebar to start chatting.")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------
# UI: render history
# ----------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----------------------
# Streaming generator using Responses API
# ----------------------

def stream_response(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Stream text deltas from the Responses API so Streamlit can render them live."""
    # Convert messages to a single input per Responses API semantics.
    # We concatenate role-tagged content; for long-term apps you may want to trim history.
    items = []
    for msg in messages:
        role = msg.get("role", "user")
        items.append({"type": "message", "role": role, "content": msg["content"]})

    try:
        with client.responses.stream(
            model=model,
            input=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                *[{"role": it["role"], "content": it["content"]} for it in items]
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ) as stream:
            for event in stream:
                # Text delta chunks
                if event.type == "response.output_text.delta":
                    yield event.delta
                # You could handle tool/function events here if you add tool use.
            # Ensure the final response is consumed to surface any server-side errors
            _final = stream.get_final_response()
    except Exception as e:
        # Surface errors inline
        yield f"\n\n**[Error]** {e}"

# ----------------------
# Handle new user input
# ----------------------
prompt = st.chat_input("Type your message…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Stream tokens as they arrive
        response_text = st.write_stream(stream_response(st.session_state.messages))

    st.session_state.messages.append({"role": "assistant", "content": response_text})
