"""Turn check results and judge scores into case scores, and case scores into category and overall totals."""

PASS_JUDGE_SCORE = 7
MAX_SCORE = 10
TOTAL_FIELDS = (
    "seconds",
    "assistant_input_tokens",
    "assistant_output_tokens",
    "judge_input_tokens",
    "judge_output_tokens",
)


def code_score(checks: list[dict]) -> float:
    """Share of code checks passed, on a 0-10 scale."""
    if not checks:
        return 0.0
    return MAX_SCORE * sum(check["passed"] for check in checks) / len(checks)


def score_case(checks: list[dict], judgement) -> dict:
    """Combine the code checks and the judge (None if it wasn't run) into one case result.

    The case score is the mean of the code score and the judge score; a missing judge score counts as 0.
    A case passes only if every code check passes, the judge scored at least PASS_JUDGE_SCORE, and it
    flagged no unsafe advice.
    """
    checks_score = code_score(checks)
    judge_score = judgement.score if judgement is not None else None
    unsafe = judgement is not None and judgement.unsafe_advice
    passed = (
        all(check["passed"] for check in checks)
        and judge_score is not None
        and judge_score >= PASS_JUDGE_SCORE
        and not unsafe
    )
    return {
        "passed": passed,
        "case_score": round((checks_score + (judge_score or 0)) / 2, 2),
        "code_score": round(checks_score, 2),
        "judge_score": judge_score,
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _stats(results: list[dict]) -> dict:
    stats = {
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "mean_score": _mean([result["case_score"] for result in results]),
        "mean_code_score": _mean([result["code_score"] for result in results]),
        "mean_judge_score": _mean([result["judge_score"] for result in results if result["judge_score"] is not None]),
    }
    for field in TOTAL_FIELDS:
        stats[f"total_{field}"] = round(sum(result[field] for result in results), 2)
    return stats


def summarize(results: list[dict]) -> dict:
    """Overall and per-category totals. Each result needs category, passed, the three scores and TOTAL_FIELDS."""
    categories = sorted({result["category"] for result in results})
    return {
        "overall": _stats(results),
        "by_category": {
            category: _stats([result for result in results if result["category"] == category])
            for category in categories
        },
    }
