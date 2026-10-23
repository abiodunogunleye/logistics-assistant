import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

from judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JUDGE_SCHEMA,
    build_prompt,
    build_system,
    clamp_score,
    judge_case,
    load_context,
)

CASE = {"id": "c1", "question": "What now?", "key_facts": ["Log it within 15 minutes", "Tell the FLM"]}
ANSWER = {"answer": "Log it.", "coverage": "full"}
SYSTEM = build_system("RUBRIC", "SOPTEXT")


def reply(data=None, text=None, stop_reason="end_turn", **usage):
    if text is None:
        text = json.dumps({"score": 8, "reason": "Good.", "unsafe_advice": False, "invented_rules": False, **(data or {})})
    usage = {"input_tokens": 100, "output_tokens": 20, **usage}
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], usage=SimpleNamespace(**usage), stop_reason=stop_reason
    )


class FakeJudgeClient:
    def __init__(self, response=None, error=None):
        self.response = response or reply()
        self.error = error
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_judge_returns_the_score_reason_and_tokens():
    client = FakeJudgeClient(reply(cache_read_input_tokens=900, cache_creation_input_tokens=0))

    result = judge_case(client, DEFAULT_JUDGE_MODEL, SYSTEM, CASE, ANSWER)

    assert (result.score, result.reason, result.error) == (8, "Good.", None)
    assert (result.input_tokens, result.output_tokens) == (1000, 20)
    assert not result.unsafe_advice and not result.invented_rules


def test_request_carries_model_cached_system_schema_and_case():
    client = FakeJudgeClient()

    judge_case(client, "judge-model", SYSTEM, CASE, ANSWER)

    call = client.calls[0]
    assert call["model"] == "judge-model"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "RUBRIC" in call["system"][0]["text"] and "SOPTEXT" in call["system"][0]["text"]
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": JUDGE_SCHEMA}
    prompt = call["messages"][0]["content"]
    assert "What now?" in prompt and "- Tell the FLM" in prompt and "Log it." in prompt


@pytest.mark.parametrize(
    "flags, score, expected",
    [
        ({"unsafe_advice": True}, 9, 3),
        ({"invented_rules": True}, 9, 6),
        ({"unsafe_advice": True, "invented_rules": True}, 9, 3),
        ({"invented_rules": True}, 4, 4),
        ({}, 9, 9),
    ],
    ids=["unsafe", "invented", "both", "already-low", "clean"],
)
def test_flags_cap_the_score(flags, score, expected):
    result = judge_case(FakeJudgeClient(reply({"score": score, **flags})), "m", SYSTEM, CASE, ANSWER)

    assert result.score == expected


def test_clamp_score_directly():
    assert clamp_score(10, True, False) == 3
    assert clamp_score(10, False, True) == 6
    assert clamp_score(2, True, True) == 2


@pytest.mark.parametrize(
    "response, message",
    [
        (reply(text="not json"), "unusable"),
        (reply({"score": 11}), "invalid fields"),
        (reply({"score": 0}), "invalid fields"),
        (reply({"score": True}), "invalid fields"),
        (reply({"unsafe_advice": "no"}), "invalid fields"),
        (reply(text=json.dumps({"score": 8})), "unusable"),
        (reply(stop_reason="refusal"), "refusal"),
        (reply(stop_reason="max_tokens"), "max_tokens"),
    ],
    ids=["not-json", "score-high", "score-zero", "score-bool", "flag-string", "missing-fields", "refusal", "max-tokens"],
)
def test_unusable_replies_give_an_error_and_no_score(response, message):
    result = judge_case(FakeJudgeClient(response), "m", SYSTEM, CASE, ANSWER)

    assert result.score is None
    assert message in result.error


def test_api_exception_is_captured_not_raised():
    result = judge_case(FakeJudgeClient(error=RuntimeError("boom")), "m", SYSTEM, CASE, ANSWER)

    assert result.score is None
    assert "boom" in result.error
    assert result.input_tokens == 0


def test_load_context_wraps_each_md_file_in_name_order(tmp_path):
    (tmp_path / "b.md").write_text("second", encoding="utf-8")
    (tmp_path / "a.md").write_text("first", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    context = load_context(tmp_path)

    assert context.index('file="a.md"') < context.index('file="b.md"')
    assert "first" in context and "second" in context and "nope" not in context


def test_build_prompt_lists_every_key_fact():
    prompt = build_prompt(CASE, ANSWER)

    assert "- Log it within 15 minutes" in prompt and "- Tell the FLM" in prompt
