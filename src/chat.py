import time
from pathlib import Path

from anthropic import Anthropic, APIError

from config import load_config

ROOT = Path(__file__).resolve().parent.parent
SOPS_DIR = ROOT / "data" / "sops"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system.md"


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


def reply_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def chat_turn(client, request_params: dict, system: str, history: list, question: str):
    """Send the question with the full history, and record both sides of the turn only if the call succeeds."""
    messages = [*history, {"role": "user", "content": question}]
    response = client.messages.create(
        **request_params,
        system=system,
        messages=messages,
    )
    history.extend([messages[-1], {"role": "assistant", "content": response.content}])
    return response


def main() -> None:
    config = load_config()
    request_params = config.request_params()
    system = build_system_prompt()
    client = Anthropic(api_key=config.api_key)
    history = []

    print("Northbank Logistics Warehouse Assistant. Type 'quit' to exit.")
    print(config.describe())

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
            response = chat_turn(client, request_params, system, history, question)
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
