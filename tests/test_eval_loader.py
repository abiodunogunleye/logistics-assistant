import json
import os
import sys
from collections import Counter

import pytest

EVALS_DIR = os.path.join(os.path.dirname(__file__), "..", "evals")
sys.path.insert(0, EVALS_DIR)

from loader import CaseError, as_list, load_cases, select_cases  # noqa: E402

CASES_FILE = os.path.join(EVALS_DIR, "warehouse", "cases.json")


def make_case(**overrides):
    case = {
        "id": "c1",
        "category": "single_sop",
        "question": "What now?",
        "expected_sop_ids": ["SOP-DG-01"],
        "expected_coverage": "full",
        "key_facts": ["Do the thing"],
    }
    case.update(overrides)
    return case


def write_cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def test_warehouse_cases_file_is_valid():
    cases = load_cases(CASES_FILE)

    assert len(cases) == 24
    assert Counter(case["category"] for case in cases) == {
        "single_sop": 6,
        "multi_sop": 4,
        "safety_injury": 3,
        "not_covered": 3,
        "followup_escalation": 2,
        "edge_case": 2,
        "adversarial": 4,
    }


def test_loads_a_valid_file(tmp_path):
    assert load_cases(write_cases(tmp_path, [make_case()]))[0]["id"] == "c1"


@pytest.mark.parametrize(
    "bad_case, message",
    [
        (make_case(id=""), "'id'"),
        (make_case(question=5), "'question'"),
        (make_case(key_facts=[]), "key_facts"),
        (make_case(key_facts=["ok", ""]), "key_facts"),
        (make_case(expected_sop_ids="SOP-DG-01"), "expected_sop_ids"),
        (make_case(expected_coverage=[]), "empty list"),
        ({k: v for k, v in make_case().items() if not k.startswith("expected_")}, "expected_"),
        ("not an object", "must be an object"),
    ],
    ids=["blank-id", "bad-question", "no-facts", "blank-fact", "sop-ids-string", "empty-list", "no-expected", "not-object"],
)
def test_rejects_invalid_cases(tmp_path, bad_case, message):
    with pytest.raises(CaseError, match=message):
        load_cases(write_cases(tmp_path, [bad_case]))


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(CaseError, match="duplicate case id 'c1'"):
        load_cases(write_cases(tmp_path, [make_case(), make_case()]))


@pytest.mark.parametrize("content", ["[]", "{}", "not json"], ids=["empty-list", "object", "bad-json"])
def test_rejects_bad_files(tmp_path, content):
    path = tmp_path / "cases.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CaseError):
        load_cases(path)


def test_select_cases_by_id_and_category():
    cases = [make_case(id="a", category="x"), make_case(id="b", category="x"), make_case(id="c", category="y")]

    assert select_cases(cases, None) == cases
    assert [c["id"] for c in select_cases(cases, ["x"])] == ["a", "b"]
    assert [c["id"] for c in select_cases(cases, ["c", "a"])] == ["a", "c"]


def test_select_cases_rejects_unknown_names():
    with pytest.raises(CaseError, match="nope"):
        select_cases([make_case()], ["nope"])


def test_as_list_wraps_single_values_and_keeps_lists():
    assert as_list("FLM") == ["FLM"]
    assert as_list(False) == [False]
    assert as_list(["none", "FLM"]) == ["none", "FLM"]
