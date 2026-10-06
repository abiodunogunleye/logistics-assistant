import pytest

import chat
from chat import SOPS_DIR, build_system_prompt, load_sops, main
from config import MAX_TOKENS


@pytest.fixture
def typed(monkeypatch):
    """Feed the chat loop a fixed sequence of lines, as if typed at the prompt."""

    def feed(*lines):
        answers = iter(lines)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    return feed


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_load_sops_wraps_every_file_in_sop_tags():
    sops = load_sops()
    files = sorted(path.name for path in SOPS_DIR.glob("*.md"))

    assert len(files) == 7
    assert sops.count("<sop file=") == len(files)
    assert sops.count("</sop>") == len(files)
    for name in files:
        assert f'<sop file="{name}">' in sops


def test_load_sops_raises_when_folder_is_empty(tmp_path):
    with pytest.raises(RuntimeError, match="No SOP files"):
        load_sops(tmp_path)


def test_system_prompt_replaces_placeholder_with_sops():
    system = build_system_prompt()

    assert "{SOPS}" not in system
    assert system.startswith("You are the Northbank Logistics Warehouse Assistant.")
    assert "<sops>\n<sop file=" in system
    for doc_id in ("SOP-DG-01", "SOP-EQ-01", "SOP-GI-01", "SOP-GO-01", "SOP-IR-01", "SOP-MH-01", "SOP-PE-01"):
        assert doc_id in system


def test_sops_are_sent_as_the_system_prompt(api_key, typed, fake_anthropic):
    typed("What do I do with a damaged carton?", "quit")

    main()

    assert fake_anthropic.calls[0]["system"] == build_system_prompt()


def test_history_grows_each_turn(api_key, typed, fake_anthropic):
    typed("First question", "Follow-up question", "quit")

    main()

    first, second = (call["messages"] for call in fake_anthropic.calls)
    assert first == [{"role": "user", "content": "First question"}]
    assert len(second) == 3
    assert second[0] == {"role": "user", "content": "First question"}
    assert second[1]["role"] == "assistant"
    assert second[1]["content"] == fake_anthropic.response.content
    assert second[2] == {"role": "user", "content": "Follow-up question"}


def test_prints_only_text_blocks_and_token_counts(api_key, typed, capsys, fake_anthropic):
    typed("What do I do with a damaged carton?", "quit")

    main()

    out = capsys.readouterr().out
    assert "Assistant: Quarantine it and tell your FLM." in out
    assert "Considering the safest procedure." not in out
    assert "[Input tokens: 12 | Output tokens: 7 | " in out


def test_default_settings_are_thinking_on_at_low_effort(api_key, typed, capsys, fake_anthropic):
    typed("What do I do with a damaged carton?", "quit")

    main()

    call = fake_anthropic.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "low"}
    assert call["max_tokens"] == MAX_TOKENS
    assert "Model: claude-sonnet-5-5 | Thinking: on | Effort: low" in capsys.readouterr().out


def test_thinking_off_sends_between_tools(api_key, typed, capsys, fake_anthropic, monkeypatch):
    monkeypatch.setenv("CLAUDE_THINKING", "off")
    monkeypatch.setenv("CLAUDE_EFFORT", "medium")
    typed("What do I do with a damaged carton?", "quit")

    main()

    call = fake_anthropic.calls[0]
    assert call["thinking"] == {"type": "between_tools"}
    assert call["output_config"] == {"effort": "medium"}
    assert "Thinking: off | Effort: medium" in capsys.readouterr().out


def test_invalid_settings_raise_before_any_api_call(api_key, fake_anthropic, monkeypatch):
    monkeypatch.setenv("CLAUDE_THINKING", "off")
    monkeypatch.setenv("CLAUDE_EFFORT", "max")

    with pytest.raises(RuntimeError, match="needs thinking on"):
        main()

    assert fake_anthropic.calls == []


def test_warns_when_reply_is_cut_off(api_key, typed, capsys, fake_anthropic):
    fake_anthropic.response.stop_reason = "max_tokens"
    typed("What do I do with a damaged carton?", "quit")

    main()

    assert "[Reply cut off at the token limit." in capsys.readouterr().out


def test_no_cut_off_warning_on_a_complete_reply(api_key, typed, capsys, fake_anthropic):
    typed("What do I do with a damaged carton?", "quit")

    main()

    out = capsys.readouterr().out
    assert "cut off" not in out
    assert "declined" not in out


def test_says_so_when_claude_declines(api_key, typed, capsys, fake_anthropic):
    fake_anthropic.response.stop_reason = "refusal"
    typed("What do I do with a damaged carton?", "quit")

    main()

    assert "[Claude declined to answer this." in capsys.readouterr().out


@pytest.mark.parametrize("command", ["quit", "QUIT", "  Quit  "])
def test_quit_exits_without_calling_the_api(api_key, typed, fake_anthropic, command):
    typed(command)

    main()

    assert fake_anthropic.calls == []


def test_blank_input_is_skipped(api_key, typed, fake_anthropic):
    typed("", "   ", "quit")

    main()

    assert fake_anthropic.calls == []


class FakeAPIError(Exception):
    pass


@pytest.fixture
def failing_api(monkeypatch, fake_anthropic):
    def fail(**kwargs):
        raise FakeAPIError("overloaded")

    monkeypatch.setattr(chat, "APIError", FakeAPIError)
    monkeypatch.setattr(fake_anthropic.messages, "create", fail)
    return fake_anthropic


def test_failed_call_leaves_history_unchanged(failing_api):
    history = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"}]

    with pytest.raises(FakeAPIError):
        chat.chat_turn(failing_api, {"model": "claude-sonnet-5-5"}, "system", history, "This call fails")

    assert history == [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"}]


def test_api_error_is_printed_and_chat_continues(api_key, typed, capsys, failing_api):
    typed("This call fails", "So does this one", "quit")

    main()

    assert capsys.readouterr().out.count("API error: overloaded") == 2


def test_chat_raises_without_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()
