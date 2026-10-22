"""Code-based checks on an assistant's JSON answer."""

import json

from loader import EXPECTED_PREFIX, as_list

SOP_IDS_FIELD = "expected_sop_ids"


def parse_answer(text: str) -> dict | None:
    """The assistant's stdout as a JSON object, or None if it isn't one."""
    try:
        answer = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return answer if isinstance(answer, dict) else None


def cited_sop_ids(answer: dict) -> list[str]:
    sources = answer.get("sources")
    if not isinstance(sources, list):
        return []
    return [source["sop_id"] for source in sources if isinstance(source, dict) and "sop_id" in source]


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": passed, "detail": detail}


def _sop_check(expected: list[str], answer: dict) -> dict:
    cited = cited_sop_ids(answer)
    if not expected:
        return _check("sop_ids", not cited, f"cited {cited}, expected none" if cited else "cited none, as expected")

    missing = [sop_id for sop_id in expected if sop_id not in cited]
    extra = [sop_id for sop_id in cited if sop_id not in expected]
    detail = f"missing {missing}" if missing else "all expected SOPs cited"
    if extra:
        detail += f"; also cited {extra}"
    return _check("sop_ids", not missing, detail)


def _field_check(field: str, acceptable: list, answer: dict) -> dict:
    value = answer.get(field)
    return _check(field, value in acceptable, f"got {value!r}, accepted {acceptable}")


def grade_answer(case: dict, answer: dict | None) -> list[dict]:
    """One check per expected_... field in the case, plus valid_json.

    expected_sop_ids must all appear in the answer's sources. Any other expected_<field> is a value or a
    list of acceptable values for the answer's <field>. If there is no valid answer, every check fails.
    """
    checks = [_check("valid_json", answer is not None, "parsed" if answer is not None else "stdout was not a JSON object")]

    names = sorted((name for name in case if name.startswith(EXPECTED_PREFIX)), key=lambda name: name != SOP_IDS_FIELD)
    for name in names:
        field = name[len(EXPECTED_PREFIX):]
        if answer is None:
            checks.append(_check(field, False, "no valid answer to check"))
        elif name == SOP_IDS_FIELD:
            checks.append(_sop_check(case[name], answer))
        else:
            checks.append(_field_check(field, as_list(case[name]), answer))
    return checks
