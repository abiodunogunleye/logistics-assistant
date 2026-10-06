import importlib
import os
import sys
from types import SimpleNamespace

import anthropic
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Modules that load .env and create an Anthropic client.
API_MODULES = ("main", "ask", "chat")
ENV_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_THINKING", "CLAUDE_EFFORT")


class RealAPICallBlocked(Exception):
    pass


class BlockedAnthropic:
    def __init__(self, *args, **kwargs):
        raise RealAPICallBlocked("Tests must not create a real Anthropic client; use the fake_anthropic fixture.")


class FakeAnthropic:
    """Stands in for the Anthropic class: records each messages.create call and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.api_key = None
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def __call__(self, api_key=None, **kwargs):
        self.api_key = api_key
        return self

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture(autouse=True)
def isolate_from_env_and_api(monkeypatch):
    """Keep every test away from the real .env file and the real Claude API."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(anthropic, "Anthropic", BlockedAnthropic)
    for module_name in API_MODULES:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "load_dotenv", lambda *args, **kwargs: False)
        monkeypatch.setattr(module, "Anthropic", BlockedAnthropic)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Replace the Anthropic client with a fake whose reply starts with a thinking block, like the real API can."""
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="Considering the safest procedure."),
            SimpleNamespace(type="text", text="Quarantine it and tell your FLM."),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        stop_reason="end_turn",
    )
    fake = FakeAnthropic(response)
    for module_name in API_MODULES:
        monkeypatch.setattr(importlib.import_module(module_name), "Anthropic", fake)
    return fake
