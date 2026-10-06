import json
import sys
from types import SimpleNamespace

import pytest

from ask import StructuredAnswer, answer_schema, main
from chat import build_system_prompt

VALID_ANSWER = {
    "answer": "Tag the empty pallet, move it to the damaged-pallet area and tell your FLM.",
    "steps": [
        "Do not use it or put it into racking.",
        "Attach a Do Not Use tag.",
        "Move it to the damaged-pallet area.",
        "Tell your FLM.",
    ],
    "sources": [{"sop_id": "SOP-EQ-01", "section": "2.2"}, {"sop_id": "SOP-EQ-01", "section": "5"}],
    "coverage": "full",
    "safety_alert": False,
    "escalate_to": "FLM",
    "confidence": "high",
}


@pytest.fixture
def ask_json(monkeypatch, fake_anthropic):
    """Run ask.py --json with the fake client replying with the given text after a thinking block."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def run(reply_text, *question):
        fake_anthropic.response.content = [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=reply_text),
        ]
        monkeypatch.setattr(sys, "argv", ["ask.py", "--json", *(question or ("Damaged", "pallet?"))])
        main()
        return fake_anthropic.calls[0]

    return run


def test_valid_json_is_printed_and_matches_the_schema(ask_json, capsys):
    ask_json(json.dumps(VALID_ANSWER))

    captured = capsys.readouterr()
    assert json.loads(captured.out) == VALID_ANSWER
    assert "Input tokens: 12" in captured.err
    assert "Model: claude-sonnet-5-5 | Thinking: on | Effort: low" in captured.err


def test_json_mode_requests_structured_output(ask_json):
    call = ask_json(json.dumps(VALID_ANSWER))

    assert call["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": answer_schema()},
    }
    assert call["thinking"] == {"type": "adaptive"}
    assert call["system"] == build_system_prompt()
    assert call["messages"] == [{"role": "user", "content": "Damaged pallet?"}]


def test_json_flag_can_go_anywhere_in_the_question(monkeypatch, fake_anthropic, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_anthropic.response.content = [SimpleNamespace(type="text", text=json.dumps(VALID_ANSWER))]
    monkeypatch.setattr(sys, "argv", ["ask.py", "Damaged", "--json", "pallet?"])

    main()

    assert fake_anthropic.calls[0]["messages"][0]["content"] == "Damaged pallet?"
    assert json.loads(capsys.readouterr().out) == VALID_ANSWER


def test_schema_meets_structured_output_rules():
    schema = answer_schema()
    objects = [schema, *schema["$defs"].values()]

    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    assert "title" not in json.dumps(schema)
    assert schema["properties"]["escalate_to"]["enum"] == ["none", "FLM", "Senior FLM", "Shift Manager"]
    assert schema["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert schema["properties"]["coverage"]["enum"] == ["full", "partial", "none"]
    assert "covered_by_sops" not in schema["properties"]


def test_schema_limits_safety_alert_to_people_at_risk():
    description = answer_schema()["properties"]["safety_alert"]["description"]

    assert description.startswith("True only when a person is injured or at immediate risk")
    assert "damaged pallet being taken out of use" in description


def test_schema_describes_the_section_format():
    description = answer_schema()["$defs"]["Source"]["properties"]["section"]["description"]

    assert "'4.2'" in description
    assert "Reporting / Escalation" in description


@pytest.mark.parametrize(
    "reply_text",
    [
        "This is not JSON.",
        '{"answer": "Cut off',
        json.dumps({**VALID_ANSWER, "escalate_to": "Supervisor"}),
        json.dumps({**VALID_ANSWER, "confidence": "certain"}),
        json.dumps({**VALID_ANSWER, "coverage": "yes"}),
        json.dumps({**{key: value for key, value in VALID_ANSWER.items() if key != "coverage"}, "covered_by_sops": True}),
        json.dumps({key: value for key, value in VALID_ANSWER.items() if key != "steps"}),
        json.dumps({**VALID_ANSWER, "extra_field": "not allowed"}),
        json.dumps({**VALID_ANSWER, "sources": [{"sop_id": "SOP-EQ-01"}]}),
    ],
    ids=[
        "not-json",
        "truncated",
        "bad-role",
        "bad-confidence",
        "bad-coverage",
        "old-coverage-field",
        "missing-field",
        "extra-field",
        "bad-source",
    ],
)
def test_invalid_reply_prints_a_clear_error_and_exits(ask_json, capsys, reply_text):
    with pytest.raises(SystemExit) as exit_info:
        ask_json(reply_text)

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert captured.out == ""
    assert "Error: Claude's reply did not match the answer schema." in captured.err


@pytest.mark.parametrize(
    "stop_reason, message",
    [
        ("refusal", "Error: Claude declined to answer this"),
        ("max_tokens", "Error: the reply was cut off at the token limit"),
    ],
)
def test_incomplete_reply_prints_a_clear_error_and_exits(ask_json, capsys, fake_anthropic, stop_reason, message):
    fake_anthropic.response.stop_reason = stop_reason

    with pytest.raises(SystemExit) as exit_info:
        ask_json(json.dumps(VALID_ANSWER))

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert captured.out == ""
    assert message in captured.err


def test_plain_mode_is_unchanged(monkeypatch, fake_anthropic, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["ask.py", "Damaged", "pallet?"])

    main()

    call = fake_anthropic.calls[0]
    assert call["output_config"] == {"effort": "low"}
    assert capsys.readouterr().out == (
        "Quarantine it and tell your FLM.\n"
        "\n"
        "Model: claude-sonnet-5-5 | Thinking: on | Effort: low\n"
        "Input tokens: 12\n"
        "Output tokens: 7\n"
    )


def test_structured_answer_round_trips():
    assert StructuredAnswer.model_validate(VALID_ANSWER).model_dump() == VALID_ANSWER
