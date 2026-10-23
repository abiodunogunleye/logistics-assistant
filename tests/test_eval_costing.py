import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

from costing import cost_usd, estimate_run, estimate_tokens, format_estimate, load_pricing  # noqa: E402

PRICING = {
    "assistant-model": {"input_per_mtok": 3, "output_per_mtok": 15},
    "judge-model": {"input_per_mtok": 5, "output_per_mtok": 25},
    "unpriced": {"input_per_mtok": None, "output_per_mtok": None},
}


def test_cost_usd_uses_per_million_prices():
    assert cost_usd(PRICING, "assistant-model", 1_000_000, 100_000) == pytest.approx(3 + 1.5)


@pytest.mark.parametrize("model", ["unpriced", "missing-model"])
def test_cost_usd_is_none_without_a_price(model):
    assert cost_usd(PRICING, model, 1000, 1000) is None


def test_estimate_tokens_scales_with_cases_and_includes_the_margin():
    one = estimate_tokens(1, 40_000, 4_000)
    twenty = estimate_tokens(20, 40_000, 4_000)

    assert one["assistant_input"] == round((10_000 + 500) * 1.2)
    assert one["judge_input"] == round((10_000 + 1_000 + 1_500) * 1.2)
    assert twenty["assistant_output"] == 20 * one["assistant_output"]


def test_estimate_run_adds_assistant_and_judge_costs():
    estimate = estimate_run(20, 40_000, 4_000, PRICING, "assistant-model", "judge-model")

    usd = estimate["usd"]
    assert usd["total"] == pytest.approx(usd["assistant"] + usd["judge"])
    assert estimate["cases"] == 20


def test_estimate_total_is_unknown_if_either_price_is_missing():
    estimate = estimate_run(20, 40_000, 4_000, PRICING, "assistant-model", "unpriced")

    assert estimate["usd"]["judge"] is None and estimate["usd"]["total"] is None
    assert "unknown" in format_estimate(estimate, "assistant-model", "unpriced")


def test_format_estimate_shows_models_calls_and_total():
    text = format_estimate(estimate_run(20, 40_000, 4_000, PRICING, "assistant-model", "judge-model"), "assistant-model", "judge-model")

    assert "20 cases (20 assistant calls + 20 judge calls)" in text
    assert "assistant-model" in text and "judge-model" in text and "total: $" in text


def test_load_pricing_reads_the_file_and_tolerates_a_missing_one(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(PRICING), encoding="utf-8")

    assert load_pricing(path) == PRICING
    assert load_pricing(tmp_path / "nope.json") == {}
