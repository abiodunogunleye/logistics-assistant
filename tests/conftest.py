import importlib
import os
import re
import sys
from types import SimpleNamespace

import anthropic
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# config is the only module that loads .env; these modules create an Anthropic client.
API_MODULES = ("main", "ask", "chat")
ENV_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_THINKING", "CLAUDE_EFFORT", "CLAUDE_STREAM")


class RealAPICallBlocked(Exception):
    pass


class BlockedAnthropic:
    def __init__(self, *args, **kwargs):
        raise RealAPICallBlocked("Tests must not create a real Anthropic client; use the fake_anthropic fixture.")


def _delta(kind, **fields):
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type=kind, **fields))


def stream_events(response):
    """The events the real SDK stream yields for a response: thinking and text arrive as separate deltas."""
    events = [SimpleNamespace(type="message_start")]
    for block in response.content:
        events.append(SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type=block.type)))
        if block.type == "thinking":
            events.append(_delta("thinking_delta", thinking=block.thinking))
        elif block.type == "text":
            events.extend(_delta("text_delta", text=chunk) for chunk in re.findall(r"\S+\s*", block.text))
        events.append(SimpleNamespace(type="content_block_stop"))
    events += [SimpleNamespace(type="message_delta"), SimpleNamespace(type="message_stop")]
    return events


class FakeStream:
    """Stands in for the SDK's MessageStream context manager."""

    def __init__(self, events, final_message, fail_after=None, error=None):
        self.events = events
        self.final_message = final_message
        self.fail_after = fail_after
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        for index, event in enumerate(self.events):
            if index == self.fail_after:
                raise self.error
            yield event

    def get_final_message(self):
        return self.final_message


class FakeAnthropic:
    """Stands in for the Anthropic class: records each request and returns a canned response.

    messages.create returns the response; messages.stream returns a FakeStream of its events.
    Set stream_error and stream_fail_after to make a stream raise after that many events.
    """

    def __init__(self, response):
        self.response = response
        self.api_key = None
        self.calls = []
        self.methods = []
        self.stream_error = None
        self.stream_fail_after = None
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def __call__(self, api_key=None, **kwargs):
        self.api_key = api_key
        return self

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        self.methods.append("create")
        return self.response

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        self.methods.append("stream")
        return FakeStream(stream_events(self.response), self.response, self.stream_fail_after, self.stream_error)


@pytest.fixture(autouse=True)
def isolate_from_env_and_api(monkeypatch):
    """Keep every test away from the real .env file and the real Claude API."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(importlib.import_module("config"), "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(anthropic, "Anthropic", BlockedAnthropic)
    for module_name in API_MODULES:
        monkeypatch.setattr(importlib.import_module(module_name), "Anthropic", BlockedAnthropic)


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
