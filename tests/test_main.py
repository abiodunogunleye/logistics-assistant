import pytest

from config import MAX_TOKENS
from main import main


def test_main_raises_without_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()


def test_main_prints_only_text_blocks(monkeypatch, capsys, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    main()

    assert fake_anthropic.api_key == "test-key"
    assert fake_anthropic.calls[0]["model"] == "claude-sonnet-5-5"
    assert fake_anthropic.calls[0]["max_tokens"] == MAX_TOKENS == 16000
    assert capsys.readouterr().out == "Quarantine it and tell your FLM.\n"


def test_main_uses_thinking_and_effort_settings(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("CLAUDE_THINKING", "on")  # Opus 5.5 can't turn thinking off
    monkeypatch.setenv("CLAUDE_EFFORT", "medium")

    main()

    call = fake_anthropic.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    "stop_reason, message",
    [("max_tokens", "[Reply cut off at the token limit.]"), ("refusal", "[Claude declined to answer this.]")],
)
def test_main_flags_incomplete_replies(monkeypatch, capsys, fake_anthropic, stop_reason, message):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_anthropic.response.stop_reason = stop_reason

    main()

    assert message in capsys.readouterr().out
