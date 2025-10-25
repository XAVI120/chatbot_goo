import os
import time
from typing import Iterable, Generator, List, Dict, Any

import streamlit as st
from openai import OpenAI

# -----------------------------
# UI — Header & Help
# -----------------------------
st.set_page_config(page_title="💬 Chatbot (OpenAI)", page_icon="💬", layout="centered")
st.title("goooooooooooooooooooooooooo Chatbot")
st.write(
    """
    Minimal yet robust Streamlit chat app using the OpenAI API with streaming, configurable parameters,
    and better session handling. Provide an API key in the sidebar to start.
    """
)

# -----------------------------
# Sidebar — Config & Secrets
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Prefer Streamlit secrets when available
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    openai_api_key = st.text_input("OpenAI API Key", value=default_key, type="password")

    # Allow model selection (include a legacy-safe default)
    # NOTE: "gpt-3.5-turbo" remains for backward compatibility; newer apps often prefer GPT-4.x/5.* models.
    model = st.selectbox(
        "Model",
        options=[
            "gpt-3.5-turbo",           # legacy compatible (Chat Completions)
            "gpt-4-turbo",             # Chat Completions
            "gpt-4o",                  # typically via Responses API; see toggle below
            "gpt-4o-mini",             # typically via Responses API; see toggle below
            "gpt-5",                   # typically via Responses API; see toggle below
        ],
        index=0,
        help="If you pick 4o/5-family, switch the Backend to 'Responses (recommended)'."
    )

    backend = st.radio(
        "API Backend",
        options=["Chat Completions (legacy)", "Responses (recommended)"],
        index=0,
        help=(
            "OpenAI recommends the Responses API for new projects. "
            "Chat Completions is kept here for compatibility."
        ),
    )

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_output_tokens = st.slider("Max output tokens", 64, 4096, 512, 64)
    seed = st.number_input("Seed (optional, -1 for none)", value=-1, step=1)

    st.divider()
    st.caption("Memory & History")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    keep_last_n = st.number_input("Keep last N turns in context", value=12, min_value=2, step=1)
    if st.button("🧹 Clear chat history"):
        st.session_state.messages = []
        st.success("History cleared.")

# Guard: require API key
if not openai_api_key:
    st.info("Add your OpenAI API key in the sidebar to continue.", icon="🗝️")
    st.stop()

# -----------------------------
# Client
# -----------------------------
client = OpenAI(api_key="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")

# -----------------------------
# Helpers
# -----------------------------
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer clearly and cite sources when relevant."
)


def _truncate_history(history: List[Dict[str, str]], n_turns: int) -> List[Dict[str, str]]:
    """Keep system + last N*2 messages (user+assistant per turn)."""
    if not history:
        return []
    # preserve the first system if present
    sys_msgs = [m for m in history if m.get("role") == "system"]
    non_sys = [m for m in history if m.get("role") != "system"]
    # keep last 2*n messages (user/assistant pairs)
    tail = non_sys[-2 * n_turns :]
    return (sys_msgs[:1] + tail) if sys_msgs else tail


def _as_openai_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": h["role"], "content": h["content"]} for h in history]


# -----------------------------
# UI — Existing messages
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input
# -----------------------------
if prompt := st.chat_input("Ask me anything…"):
    # Append system message on first run
    if not any(m.get("role") == "system" for m in st.session_state.messages):
        st.session_state.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # Store & render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Truncate history to control context growth
    st.session_state.messages = _truncate_history(st.session_state.messages, keep_last_n)

    # -----------------------------
    # Generate — two backends
    # -----------------------------
    with st.chat_message("assistant"):
        try:
            if backend == "Chat Completions (legacy)":
                # This path uses the Chat Completions API with native streaming.
                stream = client.chat.completions.create(
                    model=model,
                    messages=_as_openai_messages(st.session_state.messages),
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    seed=None if seed is None or seed < 0 else int(seed),
                    stream=True,
                )
                # Stream directly into the UI; st.write_stream supports OpenAI streams
                response_text = st.write_stream(stream)

            else:
                # Recommended path: Responses API streaming (generator adapter)
                # NOTE: The OpenAI SDK yields server-sent events; we adapt to a text generator for Streamlit.
                def response_text_stream() -> Generator[str, None, None]:
                    with client.responses.stream(
                        model=model,
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages if m["role"] != "system"]
                        ],
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        seed=None if seed is None or seed < 0 else int(seed),
                    ) as stream:
                        # Iterate over incremental text deltas if available
                        for event in stream:
                            # Conservative parse: prefer generic text extractor when available
                            if hasattr(event, "data"):
                                # Some SDK versions provide event.data with {type, delta}
                                delta = getattr(event, "delta", None) or getattr(event, "data", None)
                                if isinstance(delta, dict):
                                    chunk = (
                                        delta.get("delta")
                                        or delta.get("text")
                                        or delta.get("output_text")
                                    )
                                    if isinstance(chunk, str) and chunk:
                                        yield chunk
                        # Ensure we flush the final text if provided by helper
                        try:
                            final = stream.get_final_response()
                            if hasattr(final, "output_text") and final.output_text:
                                yield final.output_text
                        except Exception:
                            pass

                response_text = st.write_stream(response_text_stream())

            # Store assistant message
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            # Optional: collect quick feedback on the response
            fb = st.feedback("thumbs", key=f"fb_{len(st.session_state.messages)}")
            if fb is not None:
                st.toast(f"Thanks for the feedback: {fb}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Tip: Use the sidebar to switch between legacy Chat Completions and the newer Responses API, "
    "adjust temperature/token limits, and trim the conversation length to control costs."
)
