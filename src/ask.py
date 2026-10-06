import sys
from typing import Literal

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chat import build_system_prompt, reply_text
from config import load_config

DEFAULT_QUESTION = "In one sentence, what should a warehouse operative do if they find damaged stock?"
JSON_FLAG = "--json"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sop_id: str = Field(description="SOP document ID, for example SOP-DG-01.")
    section: str = Field(description="Section of that SOP the information came from, for example 'Section 2'.")


# The --json reply. Field descriptions are sent to Claude as part of the schema.
class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Short summary of the answer for the operative, in plain UK English.")
    steps: list[str] = Field(description="Steps to follow, in order. Empty if the SOPs do not cover the question.")
    sources: list[Source] = Field(description="Every SOP section the answer is based on. Empty if none.")
    covered_by_sops: bool = Field(description="True only if the provided SOPs answer the question.")
    safety_alert: bool = Field(description="True if the question involves an injury, a hazard, or a dangerous situation.")
    escalate_to: Literal["none", "FLM", "Senior FLM", "Shift Manager"] = Field(
        description="The role the operative must report this to, as named in the SOPs, or 'none' if no escalation is needed."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident you are that the answer is correct and fully supported by the SOPs."
    )


def _without_titles(schema):
    if isinstance(schema, dict):
        return {key: _without_titles(value) for key, value in schema.items() if key != "title"}
    if isinstance(schema, list):
        return [_without_titles(item) for item in schema]
    return schema


def answer_schema() -> dict:
    """StructuredAnswer as a JSON schema for structured outputs, without pydantic's title annotations."""
    return _without_titles(StructuredAnswer.model_json_schema())


def ask_json(client, config, system: str, question: str) -> bool:
    """Ask for a reply constrained to StructuredAnswer, print it as JSON, and return whether it was valid."""
    params = config.request_params()
    params["output_config"] = {
        **params["output_config"],
        "format": {"type": "json_schema", "schema": answer_schema()},
    }
    response = client.messages.create(
        **params,
        system=system,
        messages=[{"role": "user", "content": question}],
    )

    if response.stop_reason == "refusal":
        print("Error: Claude declined to answer this, so there is no JSON answer.", file=sys.stderr)
        return False
    if response.stop_reason == "max_tokens":
        print("Error: the reply was cut off at the token limit, so the JSON is incomplete.", file=sys.stderr)
        return False

    try:
        answer = StructuredAnswer.model_validate_json(reply_text(response))
    except ValidationError as error:
        print(f"Error: Claude's reply did not match the answer schema.\n{error}", file=sys.stderr)
        return False

    print(answer.model_dump_json(indent=2))
    print(config.describe(), file=sys.stderr)
    print(f"Input tokens: {response.usage.input_tokens}", file=sys.stderr)
    print(f"Output tokens: {response.usage.output_tokens}", file=sys.stderr)
    return True


def main() -> None:
    config = load_config()
    args = sys.argv[1:]
    json_mode = JSON_FLAG in args
    question = " ".join(arg for arg in args if arg != JSON_FLAG) or DEFAULT_QUESTION
    system = build_system_prompt()

    client = Anthropic(api_key=config.api_key)

    if json_mode:
        if not ask_json(client, config, system, question):
            sys.exit(1)
        return

    response = client.messages.create(
        **config.request_params(),
        system=system,
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
