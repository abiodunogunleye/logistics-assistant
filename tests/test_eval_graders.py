import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

from graders import cited_sop_ids, grade_answer, parse_answer  # noqa: E402

CASE = {
    "id": "c1",
    "expected_sop_ids": ["SOP-GI-01", "SOP-DG-01"],
    "expected_safety_alert": False,
    "expected_coverage": "full",
    "expected_escalate_to": ["none", "FLM"],
}


def answer(**overrides):
    base = {
        "sources": [{"sop_id": "SOP-GI-01", "section": "4"}, {"sop_id": "SOP-DG-01", "section": "2.3"}],
        "safety_alert": False,
        "coverage": "full",
        "escalate_to": "FLM",
    }
    base.update(overrides)
    return base


def by_name(checks):
    return {check["name"]: check for check in checks}


def test_a_correct_answer_passes_every_check():
    checks = grade_answer(CASE, answer())

    assert [check["name"] for check in checks] == ["valid_json", "sop_ids", "safety_alert", "coverage", "escalate_to"]
    assert all(check["passed"] for check in checks)


def test_missing_expected_sop_fails_and_names_it():
    checks = by_name(grade_answer(CASE, answer(sources=[{"sop_id": "SOP-GI-01", "section": "4"}])))

    assert not checks["sop_ids"]["passed"]
    assert "SOP-DG-01" in checks["sop_ids"]["detail"]


def test_extra_cited_sops_are_tolerated_but_reported():
    sources = answer()["sources"] + [{"sop_id": "SOP-IR-01", "section": "Definitions"}]

    check = by_name(grade_answer(CASE, answer(sources=sources)))["sop_ids"]

    assert check["passed"]
    assert "SOP-IR-01" in check["detail"]


def test_case_with_no_expected_sops_requires_empty_sources():
    case = {**CASE, "expected_sop_ids": []}

    assert by_name(grade_answer(case, answer(sources=[])))["sop_ids"]["passed"]
    assert not by_name(grade_answer(case, answer()))["sop_ids"]["passed"]


@pytest.mark.parametrize("value, passed", [("none", True), ("FLM", True), ("Senior FLM", False)])
def test_escalate_to_accepts_any_value_in_the_list(value, passed):
    assert by_name(grade_answer(CASE, answer(escalate_to=value)))["escalate_to"]["passed"] is passed


def test_a_single_string_expected_value_is_an_exact_match():
    case = {**CASE, "expected_escalate_to": "Senior FLM"}

    assert by_name(grade_answer(case, answer(escalate_to="Senior FLM")))["escalate_to"]["passed"]
    assert not by_name(grade_answer(case, answer(escalate_to="FLM")))["escalate_to"]["passed"]


def test_wrong_safety_alert_and_coverage_fail():
    checks = by_name(grade_answer(CASE, answer(safety_alert=True, coverage="partial")))

    assert not checks["safety_alert"]["passed"]
    assert not checks["coverage"]["passed"]
    assert "True" in checks["safety_alert"]["detail"]


def test_missing_field_in_answer_fails_without_crashing():
    checks = by_name(grade_answer(CASE, {"sources": "oops"}))

    assert checks["valid_json"]["passed"]
    assert not any(checks[name]["passed"] for name in ("sop_ids", "safety_alert", "coverage", "escalate_to"))


def test_no_answer_fails_every_check():
    checks = grade_answer(CASE, None)

    assert len(checks) == 5
    assert not any(check["passed"] for check in checks)


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "42", "null"])
def test_parse_answer_rejects_non_objects(text):
    assert parse_answer(text) is None


def test_parse_answer_accepts_a_json_object():
    assert parse_answer(json.dumps({"a": 1})) == {"a": 1}


def test_cited_sop_ids_ignores_malformed_sources():
    assert cited_sop_ids({"sources": [{"sop_id": "SOP-A"}, "x", {"section": "1"}]}) == ["SOP-A"]
    assert cited_sop_ids({}) == []
