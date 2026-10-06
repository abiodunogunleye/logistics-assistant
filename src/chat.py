import os
import time
from pathlib import Path

from anthropic import Anthropic, APIError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SOPS_DIR = ROOT / "data" / "sops"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system.md"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000

# CLAUDE_THINKING values and the thinking config each one sends.
# Sonnet 5.5 rejects {"type": "disabled"}; "between_tools" is how it turns thinking off.
THINKING_MODES = {"on": {"type": "adaptive"}, "off": {"type": "between_tools"}}
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EFFORT_LEVELS_WITHOUT_THINKING = ("low", "medium", "high")
DEFAULT_THINKING = "on"
DEFAULT_EFFORT = "low"


def load_sops(sops_dir: Path = SOPS_DIR) -> str:
    paths = sorted(sops_dir.glob("*.md"))
    if not paths:
        raise RuntimeError(f"No SOP files found in {sops_dir}.")

    return "\n\n".join(
        f'<sop file="{path.name}">\n{path.read_text(encoding="utf-8").strip()}\n</sop>' for path in paths
    )


def build_system_prompt(prompt_path: Path = SYSTEM_PROMPT_PATH, sops_dir: Path = SOPS_DIR) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    return template.replace("{SOPS}", load_sops(sops_dir))


def load_settings() -> dict:
    """Read the model, thinking and effort settings from the environment and return them as request parameters."""
    model = os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
    thinking = (os.environ.get("CLAUDE_THINKING") or DEFAULT_THINKING).strip().lower()
    effort = (os.environ.get("CLAUDE_EFFORT") or DEFAULT_EFFORT).strip().lower()

    if thinking not in THINKING_MODES:
        raise RuntimeError(f"CLAUDE_THINKING must be 'on' or 'off', not '{thinking}'.")
    if effort not in EFFORT_LEVELS:
        raise RuntimeError(f"CLAUDE_EFFORT must be one of {', '.join(EFFORT_LEVELS)}, not '{effort}'.")
    if thinking == "off" and effort not in EFFORT_LEVELS_WITHOUT_THINKING:
        raise RuntimeError(f"CLAUDE_EFFORT={effort} needs thinking on. With CLAUDE_THINKING=off, use low, medium or high.")

    return {"model": model, "thinking": THINKING_MODES[thinking], "output_config": {"effort": effort}}


def describe_settings(settings: dict) -> str:
    thinking = "on" if settings["thinking"] == THINKING_MODES["on"] else "off"
    return f"Model: {settings['model']} | Thinking: {thinking} | Effort: {settings['output_config']['effort']}"


def reply_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def chat_turn(client, settings: dict, system: str, history: list, question: str):
    """Send the question with the full history, and record both sides of the turn only if the call succeeds."""
    messages = [*history, {"role": "user", "content": question}]
    response = client.messages.create(
        **settings,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    )
    history.extend([messages[-1], {"role": "assistant", "content": response.content}])
    return response


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    settings = load_settings()
    system = build_system_prompt()
    client = Anthropic(api_key=api_key)
    history = []

    print("Northbank Logistics Warehouse Assistant. Type 'quit' to exit.")
    print(describe_settings(settings))

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() == "quit":
            break
        if not question:
            continue

        started = time.perf_counter()
        try:
            response = chat_turn(client, settings, system, history, question)
        except APIError as error:
            print(f"\nAPI error: {error}")
            continue
        elapsed = time.perf_counter() - started

        print(f"\nAssistant: {reply_text(response)}")
        if response.stop_reason == "max_tokens":
            print('\n[Reply cut off at the token limit. Ask "continue" for the rest.]')
        elif response.stop_reason == "refusal":
            print("\n[Claude declined to answer this. Rephrase the question or ask your FLM.]")
        print(
            f"\n[Input tokens: {response.usage.input_tokens} | Output tokens: {response.usage.output_tokens}"
            f" | {elapsed:.1f}s]"
        )


if __name__ == "__main__":
    main()
