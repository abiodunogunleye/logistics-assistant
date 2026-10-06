import os

import pytest

import ask
import chat
import main
from conftest import RealAPICallBlocked


@pytest.mark.parametrize("module", [main, ask, chat])
def test_real_client_is_blocked(monkeypatch, module):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["ask.py"])

    with pytest.raises(RealAPICallBlocked):
        module.main()


@pytest.mark.parametrize("module", [main, ask, chat])
def test_dotenv_is_not_loaded(monkeypatch, module):
    monkeypatch.setattr("sys.argv", ["ask.py"])

    with pytest.raises(RuntimeError):
        module.main()

    assert "ANTHROPIC_API_KEY" not in os.environ
