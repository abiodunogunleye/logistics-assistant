import pytest

from config import MAX_TOKENS, load_config


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_raises_without_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        load_config()


def test_defaults_are_sonnet_with_thinking_on_at_low_effort(api_key):
    config = load_config()

    assert config.api_key == "test-key"
    assert config.request_params() == {
        "model": "claude-sonnet-5-5",
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }
    assert config.describe() == "Model: claude-sonnet-5-5 | Thinking: on | Effort: low"


def test_blank_settings_use_defaults(api_key, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "")
    monkeypatch.setenv("CLAUDE_THINKING", "")
    monkeypatch.setenv("CLAUDE_EFFORT", "")

    assert load_config().describe() == "Model: claude-sonnet-5-5 | Thinking: on | Effort: low"


def test_reads_model_thinking_and_effort(api_key, monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("CLAUDE_THINKING", "off")
    monkeypatch.setenv("CLAUDE_EFFORT", "medium")

    params = load_config().request_params()

    assert params["model"] == "claude-opus-5-5"
    assert params["thinking"] == {"type": "between_tools"}
    assert params["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize("thinking, effort", [("ON", "HIGH"), (" on ", " xhigh "), ("on", "max")])
def test_settings_ignore_case_and_spaces(api_key, monkeypatch, thinking, effort):
    monkeypatch.setenv("CLAUDE_THINKING", thinking)
    monkeypatch.setenv("CLAUDE_EFFORT", effort)

    config = load_config()

    assert config.thinking == "on"
    assert config.effort == effort.strip().lower()


@pytest.mark.parametrize(
    "thinking, effort, message",
    [
        ("yes", "low", "CLAUDE_THINKING must be 'on' or 'off'"),
        ("on", "extreme", "CLAUDE_EFFORT must be one of"),
        ("off", "xhigh", "needs thinking on"),
        ("off", "max", "needs thinking on"),
    ],
)
def test_invalid_settings_raise(api_key, monkeypatch, thinking, effort, message):
    monkeypatch.setenv("CLAUDE_THINKING", thinking)
    monkeypatch.setenv("CLAUDE_EFFORT", effort)

    with pytest.raises(RuntimeError, match=message):
        load_config()


def test_api_key_is_kept_out_of_printed_config(api_key):
    assert "test-key" not in repr(load_config())
