"""The one place that loads .env and reads the API key, model, thinking, effort and streaming settings."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000

# CLAUDE_THINKING values and the thinking config each one sends.
# Sonnet 5.5 rejects {"type": "disabled"}; "between_tools" is how it turns thinking off.
THINKING_MODES = {"on": {"type": "adaptive"}, "off": {"type": "between_tools"}}
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EFFORT_LEVELS_WITHOUT_THINKING = ("low", "medium", "high")
DEFAULT_THINKING = "on"
DEFAULT_EFFORT = "low"
STREAM_MODES = {"on": True, "off": False}
DEFAULT_STREAM = "on"


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    model: str
    thinking: str
    effort: str
    stream: bool = True

    def request_params(self) -> dict:
        """The settings as keyword arguments for client.messages.create."""
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "thinking": THINKING_MODES[self.thinking],
            "output_config": {"effort": self.effort},
        }

    def describe(self) -> str:
        return f"Model: {self.model} | Thinking: {self.thinking} | Effort: {self.effort}"

    def describe_with_streaming(self) -> str:
        return f"{self.describe()} | Streaming: {'on' if self.stream else 'off'}"


def load_config() -> Config:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    model = os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
    thinking = (os.environ.get("CLAUDE_THINKING") or DEFAULT_THINKING).strip().lower()
    effort = (os.environ.get("CLAUDE_EFFORT") or DEFAULT_EFFORT).strip().lower()
    stream = (os.environ.get("CLAUDE_STREAM") or DEFAULT_STREAM).strip().lower()

    if thinking not in THINKING_MODES:
        raise RuntimeError(f"CLAUDE_THINKING must be 'on' or 'off', not '{thinking}'.")
    if effort not in EFFORT_LEVELS:
        raise RuntimeError(f"CLAUDE_EFFORT must be one of {', '.join(EFFORT_LEVELS)}, not '{effort}'.")
    if thinking == "off" and effort not in EFFORT_LEVELS_WITHOUT_THINKING:
        raise RuntimeError(f"CLAUDE_EFFORT={effort} needs thinking on. With CLAUDE_THINKING=off, use low, medium or high.")
    if stream not in STREAM_MODES:
        raise RuntimeError(f"CLAUDE_STREAM must be 'on' or 'off', not '{stream}'.")

    return Config(api_key=api_key, model=model, thinking=thinking, effort=effort, stream=STREAM_MODES[stream])
