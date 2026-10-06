import sys

import pytest

from ask import DEFAULT_MODEL, DEFAULT_QUESTION, main


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

    out = capsys.readouterr().out
    assert out.startswith("Quarantine it and tell your supervisor.\n")
    assert "Considering the safest procedure." not in out
    assert "Input tokens: 12" in out
    assert "Output tokens: 7" in out
