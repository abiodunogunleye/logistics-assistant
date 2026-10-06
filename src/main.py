import os

from anthropic import Anthropic
from dotenv import load_dotenv

MAX_TOKENS = 16000


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": "Hello from the logistics assistant!"}],
    )

    print("".join(block.text for block in response.content if block.type == "text"))
    if response.stop_reason == "max_tokens":
        print("\n[Reply cut off at the token limit.]")
    elif response.stop_reason == "refusal":
        print("\n[Claude declined to answer this.]")


if __name__ == "__main__":
    main()
