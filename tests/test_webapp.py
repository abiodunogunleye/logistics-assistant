import json
import threading
from datetime import date
from types import SimpleNamespace

import httpx2
import pytest

from ask import AnswerError, StructuredAnswer, answer_schema
from chat import build_system_prompt
from config import Config
from conftest import FakeAnthropic
from webapp import (
    DAILY_LIMIT_MESSAGE,
    SESSION_LIMIT,
    SESSION_LIMIT_MESSAGE,
    DailyCounter,
    admit,
    answer_so_far,
    cited_sources,
    example_questions,
    latest_eval_summary,
    stream_structured,
)

ANSWER = {
    "answer": 'Tag it "Do Not Use" and tell your FLM.\nThen move it.',
    "steps": ["Tag it.", "Tell your FLM."],
    "sources": [
        {"sop_id": "SOP-EQ-01", "section": "2.2"},
        {"sop_id": "SOP-EQ-01", "section": "5"},
        {"sop_id": "SOP-EQ-01", "section": "2.2"},
    ],
    "coverage": "full",
    "safety_alert": False,
    "escalate_to": "FLM",
    "confidence": "high",
}
CONFIG = Config(api_key="test-key", model="claude-sonnet-5-5", thinking="off", effort="low")


def make_client(reply_text, stop_reason="end_turn"):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=reply_text)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason=stop_reason,
    )
    return FakeAnthropic(response)


# answer_so_far


def test_answer_so_far_grows_as_the_json_arrives():
    full = json.dumps(ANSWER)
    shown = [answer_so_far(full[:end]) for end in range(len(full) + 1)]

    assert shown[0] == ""
    assert shown[-1] == ANSWER["answer"]
    assert all(later.startswith(earlier) for earlier, later in zip(shown, shown[1:]))


def test_answer_so_far_handles_a_split_escape_and_unicode():
    assert answer_so_far('{"answer": "a\\') == "a"
    assert answer_so_far('{"answer": "a\\u00') == "a"
    assert answer_so_far('{"answer": "caf\\u00e9 \\"x\\""') == 'café "x"'


def test_answer_so_far_ignores_a_buffer_without_the_answer_yet():
    assert answer_so_far("") == ""
    assert answer_so_far('{"ans') == ""
    assert answer_so_far('{"answer":') == ""


# stream_structured


def test_stream_structured_yields_growing_text_then_the_validated_answer():
    client = make_client(json.dumps(ANSWER))

    items = list(stream_structured(client, CONFIG, "system text", "Damaged pallet?"))

    texts = [value for kind, value in items if kind == "text"]
    assert texts[-1] == ANSWER["answer"]
    assert len(texts) > 1
    kind, answer = items[-1]
    assert kind == "done"
    assert answer == StructuredAnswer.model_validate(ANSWER)


def test_stream_structured_sends_the_shared_request_settings():
    client = make_client(json.dumps(ANSWER))

    list(stream_structured(client, CONFIG, build_system_prompt(), "Damaged pallet?"))

    call = client.calls[0]
    assert client.methods == ["stream"]
    assert call["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": answer_schema()}}
    assert call["system"] == build_system_prompt()
    assert call["messages"] == [{"role": "user", "content": "Damaged pallet?"}]


@pytest.mark.parametrize(
    ("reply", "stop_reason", "expected"),
    [
        ("", "refusal", "declined"),
        ('{"answer": "Cut', "max_tokens", "cut off"),
        ('{"answer": "No other fields"}', "end_turn", "did not match"),
    ],
)
def test_stream_structured_raises_a_clear_error_for_a_bad_reply(reply, stop_reason, expected):
    client = make_client(reply, stop_reason)

    with pytest.raises(AnswerError, match=expected):
        list(stream_structured(client, CONFIG, "system", "Question?"))


def test_stream_structured_reports_a_connection_lost_mid_answer():
    client = make_client(json.dumps(ANSWER))
    client.stream_error = httpx2.ReadError("dropped")
    client.stream_fail_after = 8

    with pytest.raises(AnswerError, match="Connection lost"):
        list(stream_structured(client, CONFIG, "system", "Question?"))


def test_stream_structured_raises_an_api_error_that_comes_before_any_text():
    client = make_client(json.dumps(ANSWER))
    client.stream_error = httpx2.ConnectError("no network")
    client.stream_fail_after = 0

    with pytest.raises(httpx2.ConnectError):
        list(stream_structured(client, CONFIG, "system", "Question?"))


def test_cited_sources_are_short_labels_without_repeats():
    answer = StructuredAnswer.model_validate(ANSWER)

    assert cited_sources(answer) == ["SOP-EQ-01 §2.2", "SOP-EQ-01 §5"]


# limits


def test_daily_counter_stops_at_the_cap_and_rolls_over_at_midnight_utc():
    today = [date(2026, 11, 10)]
    counter = DailyCounter(cap=2, today=lambda: today[0])

    assert [counter.try_take() for _ in range(3)] == [True, True, False]
    today[0] = date(2026, 11, 11)
    assert counter.try_take() is True


def test_daily_counter_never_lets_threads_take_more_than_the_cap():
    counter = DailyCounter(cap=50)
    results = []

    def take():
        results.extend(counter.try_take() for _ in range(20))

    threads = [threading.Thread(target=take) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 50


def test_admit_allows_a_question_and_uses_the_daily_count():
    counter = DailyCounter(cap=1)

    assert admit(0, counter) is None
    assert admit(0, counter) == DAILY_LIMIT_MESSAGE


def test_admit_stops_at_the_session_limit_without_using_the_daily_count():
    counter = DailyCounter(cap=1)

    assert admit(SESSION_LIMIT, counter) == SESSION_LIMIT_MESSAGE
    assert admit(SESSION_LIMIT - 1, counter) is None


# eval score


def write_result(folder, name, started_at, cases, passed, mean_score):
    data = {
        "started_at": started_at,
        "summary": {
            "overall": {"cases": cases, "passed": passed, "mean_score": mean_score},
            "by_category": {"a": {}, "b": {}},
        },
    }
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


def test_latest_eval_summary_prefers_the_run_with_most_cases_then_the_newest(tmp_path):
    write_result(tmp_path, "2026-10-23T001844_warehouse.json", "2026-10-23T00:18:44", 20, 17, 9.55)
    write_result(tmp_path, "2026-11-06T125147_warehouse.json", "2026-11-06T12:51:47", 20, 18, 9.68)
    write_result(tmp_path, "2026-11-06T161718_warehouse.json", "2026-11-06T16:17:18", 4, 4, 9.88)
    write_result(tmp_path, "2026-11-06T125147_warehouse_regrade-no-api-calls.json", "2026-11-06T12:51:47", 20, 20, 9.78)

    summary = latest_eval_summary(tmp_path)

    assert (summary.date, summary.cases, summary.passed, summary.mean_score, summary.categories) == (
        "2026-11-06",
        20,
        18,
        9.68,
        2,
    )


def test_latest_eval_summary_is_none_without_usable_results(tmp_path):
    assert latest_eval_summary(tmp_path) is None
    assert latest_eval_summary(tmp_path / "missing") is None

    (tmp_path / "2026-11-06T000000_warehouse.json").write_text("not json", encoding="utf-8")
    (tmp_path / "2026-11-07T000000_warehouse.json").write_text("{}", encoding="utf-8")

    assert latest_eval_summary(tmp_path) is None


def test_the_real_results_folder_has_a_summary():
    assert latest_eval_summary() is not None


# example questions


def test_example_questions_come_from_the_eval_cases(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "a", "question": "One?"}, {"id": "b", "question": "Two?"}]), encoding="utf-8")

    assert example_questions(cases, ids=("b", "a", "missing")) == ["Two?", "One?"]
    assert example_questions(tmp_path / "missing.json") == []


def test_the_real_example_questions_all_exist():
    assert len(example_questions()) == 4
