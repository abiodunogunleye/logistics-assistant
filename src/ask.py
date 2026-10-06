import sys

from anthropic import Anthropic

from config import load_config

DEFAULT_QUESTION = "In one sentence, what should a warehouse operative do if they find damaged stock?"


def main() -> None:
    config = load_config()
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    client = Anthropic(api_key=config.api_key)

    response = client.messages.create(
        **config.request_params(),
        messages=[{"role": "user", "content": question}],
    )

    print("".join(block.text for block in response.content if block.type == "text"))
    if response.stop_reason == "max_tokens":
        print("\n[Reply cut off at the token limit.]")
    elif response.stop_reason == "refusal":
        print("\n[Claude declined to answer this.]")
    print()
    print(config.describe())
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
