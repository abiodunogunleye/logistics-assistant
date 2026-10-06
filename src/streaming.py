"""Getting an answer from Claude and printing its text, streamed as it arrives or all at once."""

import sys
import time
from dataclasses import dataclass

import httpx2
from anthropic import APIError

# Errors that can end a stream early. The SDK wraps most network failures in APIError
# (APIConnectionError is a subclass), but a read failure mid-stream can surface from httpx2.
STREAM_ERRORS = (APIError, httpx2.HTTPError)

CONNECTION_LOST_NOTE = "[Connection lost mid-answer. The reply above is incomplete. Please ask again.]"


@dataclass
class Answer:
    message: object | None  # the final Message, or None if the connection dropped mid-answer
    first_text_seconds: float | None  # None if no text arrived
    total_seconds: float


def reply_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def get_answer(client, request_params: dict, system: str, messages: list, stream: bool) -> Answer:
    """Print the answer's text to stdout (never thinking) and return it with timings.

    With stream on, text is printed as it arrives. An error before any text is raised as
    usual; an error after text has started returns an Answer with message None, so the
    caller can say the reply is incomplete.
    """
    started = time.perf_counter()

    if not stream:
        response = client.messages.create(**request_params, system=system, messages=messages)
        print(reply_text(response), end="", flush=True)
        elapsed = time.perf_counter() - started
        return Answer(message=response, first_text_seconds=elapsed, total_seconds=elapsed)

    first_text_seconds = None
    try:
        with client.messages.stream(**request_params, system=system, messages=messages) as events:
            for event in events:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    if first_text_seconds is None:
                        first_text_seconds = time.perf_counter() - started
                    print(event.delta.text, end="", flush=True)
            message = events.get_final_message()
    except STREAM_ERRORS:
        if first_text_seconds is None:
            raise
        return Answer(message=None, first_text_seconds=first_text_seconds, total_seconds=time.perf_counter() - started)

    return Answer(message=message, first_text_seconds=first_text_seconds, total_seconds=time.perf_counter() - started)


def describe_answer(answer: Answer) -> str:
    first_text = "none" if answer.first_text_seconds is None else f"{answer.first_text_seconds:.1f}s"
    parts = [f"First text: {first_text}", f"Total: {answer.total_seconds:.1f}s"]
    if answer.message is None:
        parts.append("Tokens: unknown (connection lost)")
    else:
        parts.append(f"Input tokens: {answer.message.usage.input_tokens}")
        parts.append(f"Output tokens: {answer.message.usage.output_tokens}")
    return f"[{' | '.join(parts)}]"


def print_answer_stats(answer: Answer) -> None:
    print(describe_answer(answer), file=sys.stderr, flush=True)
