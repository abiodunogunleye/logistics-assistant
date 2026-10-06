from pathlib import Path

from anthropic import Anthropic

from config import load_config
from streaming import CONNECTION_LOST_NOTE, STREAM_ERRORS, Answer, get_answer, print_answer_stats

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


def chat_turn(client, request_params: dict, system: str, history: list, question: str, stream: bool) -> Answer:
    """Send the question with the full history and print the answer.

    Both sides of the turn are recorded only if the answer arrived complete, so a failed or
    dropped answer never leaves an unanswered question in the history.
    """
    messages = [*history, {"role": "user", "content": question}]
    answer = get_answer(client, request_params, system, messages, stream)
    if answer.message is not None:
        history.extend([messages[-1], {"role": "assistant", "content": answer.message.content}])
    return answer


def main() -> None:
    config = load_config()
    request_params = config.request_params()
    system = build_system_prompt()
    client = Anthropic(api_key=config.api_key)
    history = []

    print("Northbank Logistics Warehouse Assistant. Type 'quit' to exit.")
    print(config.describe_with_streaming())

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

        print("\nAssistant: ", end="", flush=True)
        try:
            answer = chat_turn(client, request_params, system, history, question, config.stream)
        except STREAM_ERRORS as error:
            print(f"\nAPI error: {error}")
            continue
        print()

        if answer.message is None:
            print(f"\n{CONNECTION_LOST_NOTE}")
        elif answer.message.stop_reason == "max_tokens":
            print('\n[Reply cut off at the token limit. Ask "continue" for the rest.]')
        elif answer.message.stop_reason == "refusal":
            print("\n[Claude declined to answer this. Rephrase the question or ask your FLM.]")
        print_answer_stats(answer)


if __name__ == "__main__":
    main()
