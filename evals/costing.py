"""Token and cost estimates for an eval run. Prices come from pricing.json, which the user fills in."""

import json
from pathlib import Path

CHARS_PER_TOKEN = 4
ESTIMATE_MARGIN = 1.2
ASSISTANT_OVERHEAD_TOKENS = 500  # the assistant's rules and the question, beyond the SOP text
ASSISTANT_OUTPUT_TOKENS = 500
JUDGE_CASE_TOKENS = 1500  # question, answer and key facts per judge call
JUDGE_OUTPUT_TOKENS = 300


def load_pricing(path) -> dict:
    """{model: {"input_per_mtok": number or null, "output_per_mtok": number or null}}; empty if there is no file."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cost_usd(pricing: dict, model: str, input_tokens: float, output_tokens: float) -> float | None:
    """Dollar cost, or None if the model has no usable price. Cached input is charged at the full input price."""
    prices = pricing.get(model) or {}
    input_price, output_price = prices.get("input_per_mtok"), prices.get("output_per_mtok")
    if not isinstance(input_price, (int, float)) or not isinstance(output_price, (int, float)):
        return None
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def estimate_tokens(case_count: int, context_chars: int, rubric_chars: int) -> dict:
    """Rough token use for one run: the assistant and the judge each make one call per case."""
    context_tokens = context_chars / CHARS_PER_TOKEN
    rubric_tokens = rubric_chars / CHARS_PER_TOKEN
    raw = {
        "assistant_input": case_count * (context_tokens + ASSISTANT_OVERHEAD_TOKENS),
        "assistant_output": case_count * ASSISTANT_OUTPUT_TOKENS,
        "judge_input": case_count * (context_tokens + rubric_tokens + JUDGE_CASE_TOKENS),
        "judge_output": case_count * JUDGE_OUTPUT_TOKENS,
    }
    return {name: round(tokens * ESTIMATE_MARGIN) for name, tokens in raw.items()}


def estimate_run(case_count, context_chars, rubric_chars, pricing, assistant_model, judge_model) -> dict:
    tokens = estimate_tokens(case_count, context_chars, rubric_chars)
    assistant = cost_usd(pricing, assistant_model, tokens["assistant_input"], tokens["assistant_output"])
    judge = cost_usd(pricing, judge_model, tokens["judge_input"], tokens["judge_output"])
    total = assistant + judge if assistant is not None and judge is not None else None
    return {"cases": case_count, "tokens": tokens, "usd": {"assistant": assistant, "judge": judge, "total": total}}


def format_usd(amount: float | None) -> str:
    return "unknown (no price in pricing.json)" if amount is None else f"${amount:,.2f}"


def format_estimate(estimate: dict, assistant_model: str, judge_model: str) -> str:
    tokens, usd = estimate["tokens"], estimate["usd"]
    return "\n".join(
        [
            f"Estimated cost for {estimate['cases']} cases ({estimate['cases']} assistant calls + {estimate['cases']} judge calls):",
            f"  assistant {assistant_model}: ~{tokens['assistant_input']:,} in / ~{tokens['assistant_output']:,} out"
            f"  = {format_usd(usd['assistant'])}",
            f"  judge     {judge_model}: ~{tokens['judge_input']:,} in / ~{tokens['judge_output']:,} out"
            f"  = {format_usd(usd['judge'])}",
            f"  total: {format_usd(usd['total'])}",
            "  (upper bound: ignores prompt-cache discounts, adds a 20% margin)",
        ]
    )
