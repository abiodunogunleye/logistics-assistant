"""Streamlit web demo of the warehouse assistant. Run with: streamlit run streamlit_app.py"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import anthropic  # noqa: E402
import streamlit as st  # noqa: E402

from ask import AnswerError, StructuredAnswer  # noqa: E402
from chat import build_system_prompt  # noqa: E402
from config import load_config  # noqa: E402
from streaming import STREAM_ERRORS  # noqa: E402
from webapp import (  # noqa: E402
    BADGE_COLOURS,
    DEFAULT_DAILY_CAP,
    MAX_QUESTION_CHARS,
    SESSION_LIMIT,
    SESSION_LIMIT_MESSAGE,
    DailyCounter,
    admit,
    cited_sources,
    example_questions,
    latest_eval_summary,
    stream_structured,
)

# Settings that may come from Streamlit secrets (the deployed app). Locally, config.load_config reads .env.
SECRET_NAMES = ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_THINKING", "CLAUDE_EFFORT", "DAILY_QUESTION_CAP")
NO_ANSWER_MESSAGE = "Sorry, I couldn't give an answer to that. Try rephrasing it, or ask your FLM."
UNREACHABLE_MESSAGE = "Sorry, the assistant couldn't be reached just now. Please try again in a moment."


def copy_secrets_to_environment() -> None:
    """Make Streamlit secrets visible to config.load_config, without overriding anything already set."""
    try:
        for name in SECRET_NAMES:
            if name in st.secrets:
                os.environ.setdefault(name, str(st.secrets[name]))
    except Exception:  # no secrets file: local run, settings come from .env
        pass


@st.cache_resource
def get_config():
    copy_secrets_to_environment()
    return load_config()


@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=get_config().api_key)


@st.cache_resource
def get_system_prompt() -> str:
    return build_system_prompt()


@st.cache_resource
def get_daily_counter() -> DailyCounter:
    """One counter shared by every visitor of this running app."""
    try:
        cap = int(os.environ.get("DAILY_QUESTION_CAP") or DEFAULT_DAILY_CAP)
    except ValueError:
        cap = DEFAULT_DAILY_CAP
    return DailyCounter(cap)


def render_sidebar(used: int) -> None:
    with st.sidebar:
        st.header("Warehouse Assistant")
        st.write(
            "A demo of an AI assistant that answers warehouse questions from standard operating procedures (SOPs). "
            "Northbank Logistics is a fictional company, and the SOPs are made up. "
            "Demo only: don't use it for real operating decisions."
        )

        st.subheader("Try a question")
        for question in example_questions():
            st.button(question, key=f"example-{question}", on_click=st.session_state.update, kwargs={"pending": question})

        st.caption(f"Questions left this visit: {max(SESSION_LIMIT - used, 0)} of {SESSION_LIMIT}")

        st.subheader("Latest eval score")
        summary = latest_eval_summary()
        if summary is None:
            st.caption("No eval results yet.")
        else:
            st.metric("Mean score", f"{summary.mean_score:.2f} / 10")
            st.caption(
                f"{summary.passed} of {summary.cases} cases passed, across {summary.categories} categories. "
                f"Saved results from {summary.date}, not run live."
            )


def render_details(answer: StructuredAnswer) -> None:
    """Under an answer: safety banner, steps, cited SOPs, escalation badge."""
    if answer.safety_alert:
        st.error("**Safety alert.** If someone is injured or at immediate risk, stop and alert your FLM or a first aider now.")
    if answer.steps:
        st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(answer.steps, start=1)))

    sources = cited_sources(answer)
    st.caption("Sources: " + (", ".join(sources) if sources else "no SOP covers this"))
    if answer.escalate_to != "none":
        st.badge(f"Escalate to {answer.escalate_to}", color=BADGE_COLOURS[answer.escalate_to])
    st.caption(f"SOP coverage: {answer.coverage} · Confidence: {answer.confidence}")


def render_entry(entry: dict) -> None:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        if entry["answer"] is None:
            st.warning(entry["error"])
        else:
            st.markdown(entry["answer"].answer)
            render_details(entry["answer"])


def answer_question(question: str) -> dict:
    """Stream the answer for a new question into the page and return the entry to keep in the history."""
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            answer = None
            for kind, value in stream_structured(get_client(), get_config(), get_system_prompt(), question):
                if kind == "text":
                    placeholder.markdown(value)
                else:
                    answer = value
        except AnswerError:
            placeholder.empty()
            st.warning(NO_ANSWER_MESSAGE)
            return {"question": question, "answer": None, "error": NO_ANSWER_MESSAGE}
        except STREAM_ERRORS:
            placeholder.empty()
            st.warning(UNREACHABLE_MESSAGE)
            return {"question": question, "answer": None, "error": UNREACHABLE_MESSAGE}

        placeholder.markdown(answer.answer)
        render_details(answer)
    return {"question": question, "answer": answer, "error": None}


def main() -> None:
    st.set_page_config(page_title="Warehouse Assistant", page_icon="📦")
    st.title("📦 Northbank Warehouse Assistant")
    st.caption("Demo for a fictional company. Answers come from its SOPs.")

    try:
        get_config()
    except RuntimeError:
        st.error("This demo isn't set up yet: its API key is missing.")
        st.stop()

    state = st.session_state
    state.setdefault("history", [])
    state.setdefault("used", 0)
    state.setdefault("notice", None)

    render_sidebar(state.used)
    for entry in state.history:
        render_entry(entry)

    limit_reached = state.used >= SESSION_LIMIT
    prompt = st.chat_input("Ask about a warehouse procedure", max_chars=MAX_QUESTION_CHARS, disabled=limit_reached)
    question = state.pop("pending", None) or prompt

    if question:
        notice = admit(state.used, get_daily_counter())
        state.notice = notice
        if notice is None:
            state.used += 1
            state.history.append(answer_question(question))
            st.rerun()

    if limit_reached:
        st.info(SESSION_LIMIT_MESSAGE)
    elif state.notice:
        st.info(state.notice)


main()
