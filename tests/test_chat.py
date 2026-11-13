from pathlib import Path

import httpx2
import pytest
from anthropic import APIConnectionError

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


def test_sop_and_prompt_files_are_read_as_utf8_whatever_the_locale(tmp_path, monkeypatch):
    # Windows defaults to cp1252, which would turn the UTF-8 bytes of £ into Â£. CI runs on Linux,
    # where the default is UTF-8, so check the encoding each read asks for rather than the result alone.
    real_read_text = Path.read_text
    encodings = []

    def recording_read_text(self, encoding=None, errors=None, **kwargs):
        encodings.append(encoding)
        return real_read_text(self, encoding=encoding, errors=errors, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)
    sops_dir = tmp_path / "sops"
    sops_dir.mkdir()
    (sops_dir / "values.md").write_bytes("High-value item: over £500 — ask your FLM.".encode("utf-8"))
    prompt = tmp_path / "system.md"
    prompt.write_bytes("Use plain UK English — £ not $.\n{SOPS}".encode("utf-8"))

    system = build_system_prompt(prompt, sops_dir)

    assert "over £500 — ask your FLM." in system
    assert system.startswith("Use plain UK English — £ not $.")
    assert "Â" not in system
    assert encodings == ["utf-8", "utf-8"]


def test_real_sops_keep_the_pound_sign():
    assert "over £500" in load_sops()
    assert "Â" not in load_sops()


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

    captured = capsys.readouterr()
    assert "Assistant: Quarantine it and tell your FLM." in captured.out
    assert "Considering the safest procedure." not in captured.out + captured.err
    assert "Input tokens" not in captured.out
    assert "| Input tokens: 12 | Output tokens: 7]" in captured.err


def test_default_settings_are_thinking_off_at_low_effort(api_key, typed, capsys, fake_anthropic):
    typed("What do I do with a damaged carton?", "quit")

    main()

    call = fake_anthropic.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["thinking"] == {"type": "between_tools"}
    assert call["output_config"] == {"effort": "low"}
    assert call["max_tokens"] == MAX_TOKENS
    assert "Model: claude-sonnet-5-5 | Thinking: off | Effort: low" in capsys.readouterr().out


def test_thinking_on_sends_adaptive(api_key, typed, capsys, fake_anthropic, monkeypatch):
    monkeypatch.setenv("CLAUDE_THINKING", "on")
    monkeypatch.setenv("CLAUDE_EFFORT", "medium")
    typed("What do I do with a damaged carton?", "quit")

    main()

    call = fake_anthropic.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert "Thinking: on | Effort: medium" in capsys.readouterr().out


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


@pytest.fixture
def failing_api(fake_anthropic):
    """Make every streamed request fail before any text arrives."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    fake_anthropic.stream_error = APIConnectionError(request=request)
    fake_anthropic.stream_fail_after = 0
    return fake_anthropic


def test_failed_call_leaves_history_unchanged(failing_api):
    history = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"}]

    with pytest.raises(APIConnectionError):
        chat.chat_turn(failing_api, {"model": "claude-sonnet-5-5"}, "system", history, "This call fails", stream=True)

    assert history == [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"}]


def test_api_error_is_printed_and_chat_continues(api_key, typed, capsys, failing_api):
    typed("This call fails", "So does this one", "quit")

    main()

    assert capsys.readouterr().out.count("API error: Connection error.") == 2


def test_chat_raises_without_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()
