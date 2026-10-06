from anthropic import Anthropic

from config import load_config


def main() -> None:
    config = load_config()
    client = Anthropic(api_key=config.api_key)

    response = client.messages.create(
        **config.request_params(),
        messages=[{"role": "user", "content": "Hello from the logistics assistant!"}],
    )

    print("".join(block.text for block in response.content if block.type == "text"))
    if response.stop_reason == "max_tokens":
        print("\n[Reply cut off at the token limit.]")
    elif response.stop_reason == "refusal":
        print("\n[Claude declined to answer this.]")


if __name__ == "__main__":
    main()
