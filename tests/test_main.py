import pytest

from main import main


def test_main_raises_without_api_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()


def test_main_prints_only_text_blocks(monkeypatch, capsys, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    main()

    assert fake_anthropic.api_key == "test-key"
    assert fake_anthropic.calls[0]["model"] == "claude-sonnet-5-5"
    assert capsys.readouterr().out == "Quarantine it and tell your supervisor.\n"
