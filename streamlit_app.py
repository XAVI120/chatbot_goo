def token_kwarg(model_name: str, max_tokens_val: int) -> Dict[str, Any]:
    """Return the correct max-token kwarg for Chat Completions models.
    Responses API is handled separately because different SDK versions
    may expect `max_output_tokens` vs `max_completion_tokens`.
    """
    if model_name.startswith(("gpt-4o", "gpt-5")):
        # Not used for Responses API; kept for clarity.
        return {}
    return {"max_tokens": int(max_tokens_val)}

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
    if not any(m.get("role") == "system" for m in st.session_state.messages):
        st.session_state.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages = _truncate_history(st.session_state.messages, keep_last_n)

    with st.chat_message("assistant"):
        try:
            if backend == "Chat Completions (legacy)":
                stream = client.chat.completions.create(
                    model=model,
                    messages=_as_openai_messages(st.session_state.messages),
                    temperature=temperature,
                    seed=None if seed is None or seed < 0 else int(seed),
                    **token_kwarg(model, max_output_tokens),
                    stream=True,
                )
                response_text = st.write_stream(stream)
            else:
                # Handle Responses API compatibility for SDKs expecting different token args
                try:
                    with client.responses.stream(
                        model=model,
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages if m["role"] != "system"]
                        ],
                        temperature=temperature,
                        max_completion_tokens=max_output_tokens,
                        seed=None if seed is None or seed < 0 else int(seed),
                    ) as stream:
                        for event in stream:
                            if hasattr(event, "data"):
                                delta = getattr(event, "delta", None) or getattr(event, "data", None)
                                if isinstance(delta, dict):
                                    chunk = (
                                        delta.get("delta")
                                        or delta.get("text")
                                        or delta.get("output_text")
                                    )
                                    if isinstance(chunk, str) and chunk:
                                        st.write(chunk)
                        final = stream.get_final_response()
                        response_text = getattr(final, "output_text", "")
                except TypeError:
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
                        for event in stream:
                            if hasattr(event, "data"):
                                delta = getattr(event, "delta", None) or getattr(event, "data", None)
                                if isinstance(delta, dict):
                                    chunk = (
                                        delta.get("delta")
                                        or delta.get("text")
                                        or delta.get("output_text")
                                    )
                                    if isinstance(chunk, str) and chunk:
                                        st.write(chunk)
                        final = stream.get_final_response()
                        response_text = getattr(final, "output_text", "")

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            fb = st.feedback("thumbs", key=f"fb_{len(st.session_state.messages)}")
            if fb is not None:
                st.toast(f"Thanks for the feedback: {fb}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
