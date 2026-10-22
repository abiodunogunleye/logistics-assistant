"""Load and validate eval cases."""

import json
from pathlib import Path

REQUIRED_TEXT_FIELDS = ("id", "category", "question")
EXPECTED_PREFIX = "expected_"


class CaseError(ValueError):
    pass


def as_list(value) -> list:
    """An expected value is either one value or a list of acceptable values; always return a list."""
    return list(value) if isinstance(value, list) else [value]


def validate_case(case, position: int) -> None:
    label = f"case {position}"
    if not isinstance(case, dict):
        raise CaseError(f"{label}: must be an object")

    for field in REQUIRED_TEXT_FIELDS:
        if not isinstance(case.get(field), str) or not case[field].strip():
            raise CaseError(f"{label}: '{field}' must be a non-empty string")
    label = f"case '{case['id']}'"

    facts = case.get("key_facts")
    if not isinstance(facts, list) or not facts or not all(isinstance(fact, str) and fact.strip() for fact in facts):
        raise CaseError(f"{label}: 'key_facts' must be a non-empty list of strings")

    expected = {name: value for name, value in case.items() if name.startswith(EXPECTED_PREFIX)}
    if not expected:
        raise CaseError(f"{label}: needs at least one 'expected_...' field")
    for name, value in expected.items():
        if name == "expected_sop_ids":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise CaseError(f"{label}: 'expected_sop_ids' must be a list of strings (empty if none)")
        elif isinstance(value, list) and not value:
            raise CaseError(f"{label}: '{name}' must not be an empty list")


def load_cases(path) -> list[dict]:
    """Read cases.json and check every case, so a typo fails before any API call is made."""
    try:
        cases = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CaseError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(cases, list) or not cases:
        raise CaseError(f"{path} must contain a non-empty list of cases")

    seen = set()
    for position, case in enumerate(cases, start=1):
        validate_case(case, position)
        if case["id"] in seen:
            raise CaseError(f"duplicate case id '{case['id']}'")
        seen.add(case["id"])
    return cases


def select_cases(cases: list[dict], only: list[str] | None) -> list[dict]:
    """Keep the cases whose id or category is in `only`; everything if `only` is empty."""
    if not only:
        return cases
    known = {case["id"] for case in cases} | {case["category"] for case in cases}
    unknown = [name for name in only if name not in known]
    if unknown:
        raise CaseError(f"no case id or category named: {', '.join(unknown)}")
    return [case for case in cases if case["id"] in only or case["category"] in only]
