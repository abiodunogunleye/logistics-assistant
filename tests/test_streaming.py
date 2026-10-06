import sys
from types import SimpleNamespace

import httpx2
import pytest
from anthropic import APIConnectionError

import ask
import chat
import streaming
from conftest import stream_events
from streaming import CONNECTION_LOST_NOTE, get_answer

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
MID_STREAM_ERRORS = [APIConnectionError(request=REQUEST), httpx2.ReadError("connection reset", request=REQUEST)]


def drop_after_first_text(fake_anthropic, error):
    """Make the fake stream raise error just after the first piece of text has been sent."""
    events = stream_events(fake_anthropic.response)
    text_indexes = [i for i, event in enumerate(events) if getattr(getattr(event, "delta", None), "type", None) == "text_delta"]
    fake_anthropic.stream_fail_after = text_indexes[1]
    fake_anthropic.stream_error = error


@pytest.fixture
def printed(monkeypatch):
    """Record each piece of text get_answer prints, in order."""
    pieces = []
    monkeypatch.setattr(streaming, "print", lambda text, **kwargs: pieces.append(text), raising=False)
    return pieces


def test_text_is_printed_in_order_as_it_arrives(fake_anthropic, printed):
    answer = get_answer(fake_anthropic, {"model": "claude-sonnet-5-5"}, "system", [], stream=True)

    assert fake_anthropic.methods == ["stream"]
    assert printed == ["Quarantine ", "it ", "and ", "tell ", "your ", "FLM."]
    assert answer.message is fake_anthropic.response
    assert answer.first_text_seconds is not None


def test_thinking_is_never_printed(fake_anthropic, printed, capsys):
    fake_anthropic.response.content = [
        SimpleNamespace(type="thinking", thinking="SECRET-REASONING-1"),
        SimpleNamespace(type="text", text="Tag it. "),
        SimpleNamespace(type="thinking", thinking="SECRET-REASONING-2"),
        SimpleNamespace(type="text", text="Tell your FLM."),
    ]

    get_answer(fake_anthropic, {"model": "claude-sonnet-5-5"}, "system", [], stream=True)

    assert printed == ["Tag ", "it. ", "Tell ", "your ", "FLM."]
    assert "SECRET-REASONING" not in "".join(printed) + capsys.readouterr().err


@pytest.fixture
def run_program(monkeypatch):
    """Run chat.py or plain ask.py once with one question, and return what it printed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def run(program, capsys):
        if program == "chat":
            answers = iter(["What do I do with a damaged carton?", "quit"])
            monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
            chat.main()
        else:
            monkeypatch.setattr(sys, "argv", ["ask.py", "What", "do", "I", "do", "with", "a", "damaged", "carton?"])
            ask.main()
        return capsys.readouterr()

    return run


@pytest.mark.parametrize("program", ["chat", "ask"])
def test_cut_off_warning_still_shown_when_streaming(fake_anthropic, run_program, capsys, program):
    fake_anthropic.response.stop_reason = "max_tokens"

    captured = run_program(program, capsys)

    assert fake_anthropic.methods == ["stream"]
    assert "Quarantine it and tell your FLM." in captured.out
    assert "[Reply cut off at the token limit." in captured.out


@pytest.mark.parametrize("error", MID_STREAM_ERRORS, ids=["api-connection-error", "httpx-read-error"])
def test_mid_stream_drop_in_chat_keeps_partial_text_and_history_clean(
    fake_anthropic, monkeypatch, capsys, error
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    drop_after_first_text(fake_anthropic, error)
    answers = iter(["First question", "Second question", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    chat.main()

    captured = capsys.readouterr()
    assert "Assistant: Quarantine " in captured.out
    assert "tell your FLM." not in captured.out
    assert captured.out.count(CONNECTION_LOST_NOTE) == 2
    assert captured.err.count("Tokens: unknown (connection lost)") == 2
    # The dropped first answer was not added to the history, so the second request has only its own question.
    assert fake_anthropic.calls[1]["messages"] == [{"role": "user", "content": "Second question"}]


@pytest.mark.parametrize("error", MID_STREAM_ERRORS, ids=["api-connection-error", "httpx-read-error"])
def test_mid_stream_drop_in_ask_prints_note_and_exits(fake_anthropic, monkeypatch, capsys, error):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py", "Damaged", "carton?"])
    drop_after_first_text(fake_anthropic, error)

    with pytest.raises(SystemExit) as exit_info:
        ask.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert captured.out.startswith("Quarantine \n")
    assert CONNECTION_LOST_NOTE in captured.out
    assert "Tokens: unknown (connection lost)" in captured.err
