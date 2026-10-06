import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_QUESTION = "In one sentence, what should a warehouse operative do if they find damaged stock?"


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    model = os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )

    print("".join(block.text for block in response.content if block.type == "text"))
    print()
    print(f"Model: {model}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
