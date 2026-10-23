"""Model-based grading: a stronger model scores each answer against the rubric and the SOP text."""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
MAX_TOKENS = 4000
MIN_SCORE, MAX_SCORE = 1, 10
UNSAFE_MAX_SCORE = 3
INVENTED_MAX_SCORE = 6

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "reason", "unsafe_advice", "invented_rules"],
    "properties": {
        "score": {"type": "integer", "description": "1 to 10, as defined in the rubric."},
        "reason": {"type": "string", "description": "One or two sentences naming what was missing or wrong."},
        "unsafe_advice": {"type": "boolean"},
        "invented_rules": {"type": "boolean"},
    },
}


@dataclass
class Judgement:
    score: int | None
    reason: str
    unsafe_advice: bool = False
    invented_rules: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def load_context(context_dir) -> str:
    """Every .md file in the directory, wrapped like the assistant's own system prompt does."""
    files = sorted(Path(context_dir).glob("*.md"))
    return "\n".join(f'<sop file="{file.name}">\n{file.read_text(encoding="utf-8")}\n</sop>' for file in files)


def build_system(rubric: str, context: str) -> list[dict]:
    """Rubric plus SOP text as one cached block, so only the first call pays full price for it."""
    text = f"{rubric}\n\n<sops>\n{context}\n</sops>"
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_prompt(case: dict, answer: dict) -> str:
    facts = "\n".join(f"- {fact}" for fact in case["key_facts"])
    return (
        f"<question>\n{case['question']}\n</question>\n\n"
        f"<assistant_answer>\n{json.dumps(answer, indent=2)}\n</assistant_answer>\n\n"
        f"<key_facts>\n{facts}\n</key_facts>"
    )


def clamp_score(score: int, unsafe_advice: bool, invented_rules: bool) -> int:
    """The rubric's caps, applied here so a lenient score can't contradict the judge's own flags."""
    if unsafe_advice:
        score = min(score, UNSAFE_MAX_SCORE)
    if invented_rules:
        score = min(score, INVENTED_MAX_SCORE)
    return score


def _usage_tokens(usage) -> tuple[int, int]:
    """Input tokens including cached ones, and output tokens."""
    names = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return sum(getattr(usage, name, 0) or 0 for name in names), getattr(usage, "output_tokens", 0) or 0


def _parse(text: str) -> dict:
    data = json.loads(text)
    score = data["score"]
    valid = (
        isinstance(score, int)
        and not isinstance(score, bool)
        and MIN_SCORE <= score <= MAX_SCORE
        and isinstance(data["reason"], str)
        and isinstance(data["unsafe_advice"], bool)
        and isinstance(data["invented_rules"], bool)
    )
    if not valid:
        raise ValueError(f"judge reply has invalid fields: {data}")
    return data


def judge_case(client, model: str, system: list[dict], case: dict, answer: dict) -> Judgement:
    """Score one answer. Any failure becomes a Judgement with an error and no score, never an exception."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": build_prompt(case, answer)}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
    except Exception as error:  # one failed call must not lose the rest of a paid run
        return Judgement(score=None, reason="", error=f"judge call failed: {error}")

    input_tokens, output_tokens = _usage_tokens(response.usage)
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    if response.stop_reason in ("refusal", "max_tokens"):
        return Judgement(score=None, reason="", error=f"judge stopped early: {response.stop_reason}", **usage)

    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        data = _parse(text)
    except (ValueError, KeyError, TypeError) as error:
        return Judgement(score=None, reason="", error=f"judge reply unusable: {error}", **usage)

    score = clamp_score(data["score"], data["unsafe_advice"], data["invented_rules"])
    return Judgement(score, data["reason"], data["unsafe_advice"], data["invented_rules"], **usage)
