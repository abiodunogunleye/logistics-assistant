import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FakeAnthropic
from webapp import SESSION_LIMIT, example_questions

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

ANSWER = {
    "answer": "Tag it and tell your FLM.",
    "steps": ["Attach a Do Not Use tag.", "Tell your FLM."],
    "sources": [{"sop_id": "SOP-EQ-01", "section": "2.2"}, {"sop_id": "SOP-DG-01", "section": "4"}],
    "coverage": "full",
    "safety_alert": False,
    "escalate_to": "FLM",
    "confidence": "high",
}


@pytest.fixture
def fake_client(monkeypatch):
    """The app's Anthropic client replaced by the fake, with the key set and Streamlit's caches emptied."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(ANSWER))],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    fake = FakeAnthropic(response)
    monkeypatch.setattr(anthropic, "Anthropic", fake)
    st.cache_resource.clear()
    yield fake
    st.cache_resource.clear()


def reply_with(fake, **changes):
    fake.response.content = [SimpleNamespace(type="text", text=json.dumps({**ANSWER, **changes}))]


def ask(app, question):
    app.chat_input[0].set_value(question).run()
    assert not app.exception
    return app


def new_app():
    app = AppTest.from_file(APP, default_timeout=10).run()
    assert not app.exception
    return app


def texts(elements):
    return [element.value for element in elements]


def test_an_answer_shows_its_text_steps_sources_and_escalation_badge(fake_client):
    app = ask(new_app(), "Damaged pallet?")

    assert "Tag it and tell your FLM." in texts(app.markdown)
    assert any("Attach a Do Not Use tag." in value for value in texts(app.markdown))
    assert "Sources: SOP-EQ-01 §2.2, SOP-DG-01 §4" in texts(app.caption)
    assert any("Escalate to FLM" in value for value in texts(app.markdown))
    assert len(app.error) == 0
    assert fake_client.calls[0]["messages"] == [{"role": "user", "content": "Damaged pallet?"}]


def test_no_escalation_badge_when_nothing_needs_escalating(fake_client):
    reply_with(fake_client, escalate_to="none", sources=[])

    app = ask(new_app(), "What's for lunch?")

    assert not any("Escalate to" in value for value in texts(app.markdown))
    assert "Sources: no SOP covers this" in texts(app.caption)


def test_a_safety_alert_shows_the_red_banner(fake_client):
    reply_with(fake_client, safety_alert=True, escalate_to="Shift Manager")

    app = ask(new_app(), "Someone's been hit by a forklift.")

    assert len(app.error) == 1
    assert "Safety alert" in app.error[0].value
    assert any("Escalate to Shift Manager" in value for value in texts(app.markdown))


def test_the_session_limit_stops_the_eleventh_question_without_calling_the_api(fake_client):
    app = new_app()
    for number in range(SESSION_LIMIT):
        ask(app, f"Question {number}?")

    assert len(fake_client.calls) == SESSION_LIMIT
    assert app.chat_input[0].disabled
    assert any("demo questions" in info.value for info in app.info)

    app.sidebar.button[0].click().run()
    assert len(fake_client.calls) == SESSION_LIMIT


def test_the_daily_cap_is_shared_across_visitors_and_shows_a_friendly_message(fake_client, monkeypatch):
    monkeypatch.setenv("DAILY_QUESTION_CAP", "2")
    first, second = new_app(), new_app()

    ask(first, "One?")
    ask(second, "Two?")
    ask(first, "Three?")

    assert len(fake_client.calls) == 2
    assert any("come back tomorrow" in info.value for info in first.info)


def test_an_example_button_asks_its_question(fake_client):
    app = new_app()
    questions = example_questions()
    assert [button.label for button in app.sidebar.button] == questions

    app.sidebar.button[2].click().run()

    assert not app.exception
    assert fake_client.calls[0]["messages"][0]["content"] == questions[2]


def test_the_sidebar_says_it_is_a_fictional_demo_and_shows_the_saved_eval_score(fake_client):
    app = new_app()

    sidebar_text = " ".join(texts(app.sidebar.markdown) + texts(app.sidebar.caption))
    assert "fictional" in sidebar_text
    assert "not run live" in sidebar_text
    assert len(app.sidebar.metric) == 1
    assert fake_client.calls == []


def test_a_refused_answer_shows_a_friendly_warning(fake_client):
    fake_client.response.stop_reason = "refusal"

    app = ask(new_app(), "Something odd?")

    assert any("couldn't give an answer" in warning.value for warning in app.warning)


def test_without_an_api_key_the_app_says_it_is_not_set_up(fake_client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    app = new_app()

    assert any("isn't set up" in error.value for error in app.error)
    assert len(app.chat_input) == 0
    assert fake_client.calls == []
