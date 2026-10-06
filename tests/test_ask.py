import sys

import pytest

from ask import DEFAULT_QUESTION, main
from chat import build_system_prompt
from config import DEFAULT_MODEL, MAX_TOKENS


def test_ask_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ask.py"])

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()


def test_ask_uses_defaults(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py"])

    main()

    call = fake_anthropic.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["messages"] == [{"role": "user", "content": DEFAULT_QUESTION}]


def test_ask_sends_the_sop_system_prompt(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py", "What", "do", "I", "do", "with", "a", "damaged", "pallet?"])

    main()

    system = fake_anthropic.calls[0]["system"]
    assert system == build_system_prompt()
    assert system.startswith("You are the Northbank Logistics Warehouse Assistant.")
    assert '<sop file="damaged-goods.md">' in system


def test_ask_treats_blank_model_as_default(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "")
    monkeypatch.setattr(sys, "argv", ["ask.py"])

    main()

    assert fake_anthropic.calls[0]["model"] == DEFAULT_MODEL


def test_ask_uses_model_and_question_from_inputs(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5-5")
    monkeypatch.setattr(sys, "argv", ["ask.py", "Where", "is", "NB-1003?"])

    main()

    call = fake_anthropic.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["messages"] == [{"role": "user", "content": "Where is NB-1003?"}]


def test_ask_prints_text_and_token_counts(monkeypatch, capsys, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py"])

    main()

    captured = capsys.readouterr()
    assert captured.out == "Quarantine it and tell your FLM.\n"
    assert "Considering the safest procedure." not in captured.err
    assert "Input tokens: 12 | Output tokens: 7" in captured.err


def test_ask_sends_shared_defaults(monkeypatch, capsys, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py"])

    main()

    call = fake_anthropic.calls[0]
    assert call["max_tokens"] == MAX_TOKENS == 16000
    assert call["thinking"] == {"type": "between_tools"}
    assert call["output_config"] == {"effort": "low"}
    assert "Model: claude-sonnet-5-5 | Thinking: off | Effort: low | Streaming: on" in capsys.readouterr().err


def test_ask_uses_thinking_and_effort_settings(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_THINKING", "on")
    monkeypatch.setenv("CLAUDE_EFFORT", "high")
    monkeypatch.setattr(sys, "argv", ["ask.py"])

    main()

    call = fake_anthropic.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}


@pytest.mark.parametrize(
    "stop_reason, message",
    [("max_tokens", "[Reply cut off at the token limit.]"), ("refusal", "[Claude declined to answer this.]")],
)
def test_ask_flags_incomplete_replies(monkeypatch, capsys, fake_anthropic, stop_reason, message):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py"])
    fake_anthropic.response.stop_reason = stop_reason

    main()

    assert message in capsys.readouterr().out
