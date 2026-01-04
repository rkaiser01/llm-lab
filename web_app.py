from __future__ import annotations

from datetime import datetime
import streamlit as st
from llm_client import AVAILABLE_MODELS, get_default_model
from orchestrator import orchestrated_chat
from usage_tracker import format_usage_summary, reset_usage



def ts() -> str:
    """
    Return a formatted timestamp string.
    """
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def init_session_state() -> None:
    """
    Initialize Streamlit session state variables if they do not exist yet.
    """
    if "primary_model" not in st.session_state:
        st.session_state.primary_model = get_default_model()
    if "advisor_model" not in st.session_state:
        st.session_state.advisor_model = "o3-mini"
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts with keys: role, content, timestamp


def main() -> None:
    """
    Entry point for the Streamlit web application.
    """
    st.set_page_config(page_title="LLM Lab Orchestrator", layout="wide")
    init_session_state()

    st.title("LLM Lab – Orchestrated Web Interface")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Primary model selection
    model_names = list(AVAILABLE_MODELS.keys())
    default_index = model_names.index(st.session_state.primary_model)
    selected_primary_model = st.sidebar.selectbox(
        "Primary model",
        options=model_names,
        index=default_index,
        format_func=lambda name: f"{name} – {AVAILABLE_MODELS[name]}",
    )
    st.session_state.primary_model = selected_primary_model

    # Advisor model (fixed list for now)
    advisor_model = st.sidebar.selectbox(
        "Advisor model",
        options=["o3-mini"],
        index=0,
    )
    st.session_state.advisor_model = advisor_model

    if st.sidebar.button("Reset usage counters"):
        reset_usage()
        st.sidebar.success("Usage counters were reset.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("All times are local system time.")

    # Main input area
    st.subheader("Chat")

    user_prompt = st.text_area(
        "Your prompt",
        value="Describe a small LLM prototype idea for legal tech.",
        height=150,
    )

    col_send, col_clear = st.columns([1, 1])

    with col_send:
        send_clicked = st.button("Send", type="primary")
    with col_clear:
        clear_clicked = st.button("Clear conversation")

    if clear_clicked:
        st.session_state.messages = []
        st.success("Conversation cleared.")

    if send_clicked and user_prompt.strip():
        timestamp = ts()
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt.strip(), "timestamp": timestamp}
        )

        try:
            final_answer, advisor_notes = orchestrated_chat(
                user_prompt=user_prompt.strip(),
                primary_model=st.session_state.primary_model,
                advisor_model=st.session_state.advisor_model,
                return_advisor=True,
            )
        except Exception as exc:
            st.error(f"{ts()} ERROR > {exc}")
        else:
            advisor_timestamp = ts()
            assistant_timestamp = ts()

            st.session_state.messages.append(
                {
                    "role": "advisor",
                    "content": advisor_notes,
                    "timestamp": advisor_timestamp,
                }
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "timestamp": assistant_timestamp,
                }
            )

    # Conversation display
    st.subheader("Conversation")

    if not st.session_state.messages:
        st.info("No messages yet. Enter a prompt above and click 'Send'.")
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"]

            if role == "user":
                st.markdown(f"**{timestamp} USER INPUT**")
                st.markdown(f"> {content}")
            elif role == "advisor":
                with st.expander(f"{timestamp} ADVISOR NOTES", expanded=False):
                    st.markdown(content)
            elif role == "assistant":
                st.markdown(f"**{timestamp} ASSISTANT**")
                st.markdown(content)
            else:
                st.markdown(f"{timestamp} {role.upper()}")
                st.markdown(content)

    # Usage summary
    st.subheader("Usage and estimated cost")
    usage_text = format_usage_summary()
    st.code(f"{ts()} USAGE\n{usage_text}", language="text")


if __name__ == "__main__":
    main()
