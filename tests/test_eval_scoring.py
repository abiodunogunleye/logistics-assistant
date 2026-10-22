import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

from scoring import code_score, score_case, summarize  # noqa: E402


def checks(*passed):
    return [{"name": f"c{i}", "passed": value, "detail": ""} for i, value in enumerate(passed)]


def judged(score, unsafe=False):
    return SimpleNamespace(score=score, unsafe_advice=unsafe)


def result(category="a", passed=True, case_score=8.0, code=10.0, judge=6, **totals):
    base = {field: 0 for field in ("seconds", "assistant_input_tokens", "assistant_output_tokens")}
    base.update(judge_input_tokens=0, judge_output_tokens=0)
    return {
        "category": category,
        "passed": passed,
        "case_score": case_score,
        "code_score": code,
        "judge_score": judge,
        **base,
        **totals,
    }


def test_code_score_is_the_share_of_checks_passed_out_of_ten():
    assert code_score(checks(True, True, True, True)) == 10
    assert code_score(checks(True, False, True, False)) == 5
    assert code_score([]) == 0


def test_case_score_is_the_mean_of_code_and_judge_scores():
    scored = score_case(checks(True, True, False, False), judged(9))

    assert (scored["code_score"], scored["judge_score"], scored["case_score"]) == (5, 9, 7)


def test_passes_only_with_all_checks_and_a_judge_score_of_seven_or_more():
    assert score_case(checks(True, True), judged(7))["passed"]
    assert not score_case(checks(True, True), judged(6))["passed"]
    assert not score_case(checks(True, False), judged(10))["passed"]


def test_unsafe_advice_fails_even_a_high_score():
    assert not score_case(checks(True), judged(9, unsafe=True))["passed"]


def test_no_judge_means_no_pass_and_a_zero_judge_share():
    scored = score_case(checks(True, True), None)

    assert scored["passed"] is False
    assert scored["judge_score"] is None
    assert scored["case_score"] == 5


def test_an_unparseable_answer_scores_zero():
    scored = score_case(checks(False, False, False), None)

    assert scored["case_score"] == 0 and not scored["passed"]


@pytest.mark.parametrize("judge_score, expected", [(7, True), (10, True), (3, False)])
def test_pass_threshold(judge_score, expected):
    assert score_case(checks(True), judged(judge_score))["passed"] is expected


def test_summarize_groups_by_category_and_totals_everything():
    results = [
        result("a", True, 9.0, 10.0, 8, seconds=10, assistant_input_tokens=100, assistant_output_tokens=10),
        result("a", False, 3.0, 6.0, None, seconds=20, assistant_input_tokens=100, assistant_output_tokens=30),
        result("b", True, 7.0, 10.0, 7, seconds=5, judge_input_tokens=50, judge_output_tokens=5),
    ]

    summary = summarize(results)

    assert list(summary["by_category"]) == ["a", "b"]
    assert summary["by_category"]["a"]["cases"] == 2
    assert summary["by_category"]["a"]["passed"] == 1
    assert summary["by_category"]["a"]["mean_score"] == 6.0
    assert summary["by_category"]["a"]["mean_judge_score"] == 8.0
    assert summary["overall"]["cases"] == 3
    assert summary["overall"]["passed"] == 2
    assert summary["overall"]["mean_score"] == 6.33
    assert summary["overall"]["total_seconds"] == 35
    assert summary["overall"]["total_assistant_input_tokens"] == 200
    assert summary["overall"]["total_assistant_output_tokens"] == 40
    assert summary["overall"]["total_judge_input_tokens"] == 50


def test_summarize_handles_a_category_with_no_judge_scores():
    summary = summarize([result("a", False, 0, 0, None)])

    assert summary["overall"]["mean_judge_score"] is None
