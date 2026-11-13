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
    CASES_PATH,
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
    suite_case_ids,
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


SUITE_IDS = ["single-01", "safety-01", "adversarial-01", "adversarial-02"]


@pytest.fixture
def suite(tmp_path):
    """A cases file for a four-case suite; returns its path."""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": case_id, "question": "?"} for case_id in SUITE_IDS]), encoding="utf-8")
    return path


def write_result(folder, name, started_at, case_ids, passed, mean_score, regrade=False):
    data = {
        "started_at": started_at,
        "summary": {
            "overall": {"cases": len(case_ids), "passed": passed, "mean_score": mean_score},
            "by_category": {case_id.split("-")[0]: {} for case_id in case_ids},
        },
        "cases": [{"id": case_id} for case_id in case_ids],
    }
    if regrade:
        data["regrade"] = {"from": "earlier run"}
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(json.dumps(data), encoding="utf-8")


def test_latest_eval_summary_is_the_newest_full_run(tmp_path, suite):
    results = tmp_path / "results"
    write_result(results, "2026-11-13T063922_warehouse.json", "2026-11-13T06:39:22", SUITE_IDS, 2, 9.58)
    write_result(results, "2026-11-13T080512_warehouse.json", "2026-11-13T08:05:12", SUITE_IDS, 3, 9.71)
    write_result(results, "2026-11-13T073043_warehouse.json", "2026-11-13T07:30:43", SUITE_IDS, 4, 9.62)

    summary = latest_eval_summary(results, suite)

    assert (summary.date, summary.cases, summary.passed, summary.mean_score, summary.categories) == (
        "2026-11-13",
        4,
        3,
        9.71,
        3,
    )


def test_newer_partial_runs_and_regrades_are_not_shown(tmp_path, suite):
    results = tmp_path / "results"
    write_result(results, "2026-11-13T080512_warehouse.json", "2026-11-13T08:05:12", SUITE_IDS, 3, 9.71)
    # --only runs: newer, but they skip cases.
    write_result(results, "2026-11-13T090000_warehouse.json", "2026-11-13T09:00:00", SUITE_IDS[2:], 2, 10.0)
    write_result(results, "2026-11-13T091000_warehouse.json", "2026-11-13T09:10:00", SUITE_IDS[:3], 3, 10.0)
    # A newer regrade of the full run, saved under a plain _warehouse.json name.
    write_result(results, "2026-11-13T092000_warehouse.json", "2026-11-13T09:20:00", SUITE_IDS, 4, 9.9, regrade=True)

    summary = latest_eval_summary(results, suite)

    assert (summary.passed, summary.mean_score) == (3, 9.71)


def test_runs_from_before_cases_were_added_are_not_full(tmp_path, suite):
    results = tmp_path / "results"
    write_result(results, "2026-11-06T125147_warehouse.json", "2026-11-06T12:51:47", SUITE_IDS[:2], 2, 9.68)

    assert latest_eval_summary(results, suite) is None


def test_latest_eval_summary_is_none_without_usable_results(tmp_path, suite):
    assert latest_eval_summary(tmp_path / "missing", suite) is None
    assert latest_eval_summary(tmp_path, tmp_path / "no-cases.json") is None

    (tmp_path / "2026-11-06T000000_warehouse.json").write_text("not json", encoding="utf-8")
    (tmp_path / "2026-11-07T000000_warehouse.json").write_text("{}", encoding="utf-8")

    assert latest_eval_summary(tmp_path, suite) is None


def test_the_real_app_shows_a_full_run_of_the_whole_suite():
    # Fails if cases are added without a new full run being committed.
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    categories = {case["category"] for case in cases}

    summary = latest_eval_summary()

    assert summary is not None
    assert summary.cases == len(suite_case_ids()) == len(cases)
    assert summary.categories == len(categories)
    assert "adversarial" in categories


# example questions


def test_example_questions_come_from_the_eval_cases(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "a", "question": "One?"}, {"id": "b", "question": "Two?"}]), encoding="utf-8")

    assert example_questions(cases, ids=("b", "a", "missing")) == ["Two?", "One?"]
    assert example_questions(tmp_path / "missing.json") == []


def test_the_real_example_questions_all_exist():
    assert len(example_questions()) == 4
