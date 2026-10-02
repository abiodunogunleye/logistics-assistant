import pytest

from main import main


def test_main_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main()
