import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

import runner  # noqa: E402

WHEN = datetime(2026, 10, 22, 15, 30, 5)
GOOD_ANSWER = {
    "answer": "Quarantine it.",
    "sources": [{"sop_id": "SOP-DG-01", "section": "2"}],
    "coverage": "full",
    "safety_alert": False,
    "escalate_to": "FLM",
}


def case(case_id, category="a", **overrides):
    return {
        "id": case_id, "category": category, "question": f"Question {case_id}?",
        "expected_sop_ids": ["SOP-DG-01"], "expected_safety_alert": False, "expected_coverage": "full",
        "expected_escalate_to": ["none", "FLM"], "key_facts": ["Quarantine it"], **overrides,
    }


class FakeJudgeClient:
    def __init__(self, score=9):
        self.calls = []
        self.score = score
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps({"score": self.score, "reason": "Fine.", "unsafe_advice": False, "invented_rules": False})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=300, output_tokens=40), stop_reason="end_turn",
        )


class Env:
    """A suite on disk plus fakes for every outside call, so a run touches no network and no real subprocess."""

    def __init__(self, tmp_path, cases):
        self.suite = tmp_path / "suite"
        self.suite.mkdir()
        (self.suite / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
        (self.suite / "rubric.md").write_text("RUBRIC", encoding="utf-8")
        self.pricing = tmp_path / "pricing.json"
        self.set_prices({"assistant-m": {"input_per_mtok": 3, "output_per_mtok": 15},
                         "judge-m": {"input_per_mtok": 5, "output_per_mtok": 25}})
        self.results = tmp_path / "results"
        self.asked, self.assistant_calls, self.client = [], [], FakeJudgeClient()
        self.stdout = json.dumps(GOOD_ANSWER)
        self.clients_made = 0

    def set_prices(self, prices):
        self.pricing.write_text(json.dumps(prices), encoding="utf-8")

    def _assistant(self, question):
        self.assistant_calls.append(question)
        return {"stdout": self.stdout, "stderr": "", "returncode": 0, "seconds": 1.5,
                "input_tokens": 1000, "output_tokens": 100, "error": None}

    def _client(self):
        self.clients_made += 1
        return self.client

    def run(self, *extra, reply="y", interactive=True):
        argv = [str(self.suite), "--cmd", "unused", "--assistant-model", "assistant-m", "--judge-model", "judge-m",
                "--pricing", str(self.pricing), "--results-dir", str(self.results), *extra]
        def input_fn(prompt):
            self.asked.append(prompt)
            return reply
        return runner.main(argv, run_assistant=self._assistant, client_factory=self._client, input_fn=input_fn,
                           now=lambda: WHEN, is_interactive=lambda: interactive)

    def saved(self):
        files = list(self.results.glob("*.json"))
        assert len(files) == 1
        return files[0].name, json.loads(files[0].read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def not_in_ci(monkeypatch):
    monkeypatch.delenv("CI", raising=False)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path, [case("c1", "a"), case("c2", "a"), case("c3", "b")])


def test_full_run_grades_scores_and_saves_a_dated_file(env, capsys):
    assert env.run() == 0

    name, report = env.saved()
    assert name == "2026-10-22T153005_suite.json"
    assert report["summary"]["overall"]["passed"] == 3
    assert report["summary"]["by_category"]["a"]["cases"] == 2
    first = report["cases"][0]
    assert first["passed"] and first["case_score"] == 9.5 and first["seconds"] == 1.5
    assert (first["assistant_input_tokens"], first["judge_input_tokens"]) == (1000, 300)
    assert report["config"]["assistant_model"] == "assistant-m"
    assert len(env.client.calls) == 3
    out = capsys.readouterr().out
    assert "Overall: 3/3 passed" in out and "Saved" in out


def test_shows_the_estimate_and_asks_before_any_call(env, capsys):
    env.run(reply="n")

    out = capsys.readouterr().out
    assert "Estimated cost for 3 cases" in out
    assert env.asked == ["Run? [y/N] "]


@pytest.mark.parametrize("reply", ["n", "", "yes", "no"])
def test_anything_but_y_aborts_with_no_calls(env, capsys, reply):
    assert env.run(reply=reply) == 1
    assert env.assistant_calls == [] and env.client.calls == [] and env.clients_made == 0
    assert not env.results.exists()
    assert "No API calls were made" in capsys.readouterr().out


def test_refuses_in_ci(env, monkeypatch):
    monkeypatch.setenv("CI", "true")

    assert env.run() == 2
    assert env.asked == [] and env.assistant_calls == [] and env.clients_made == 0


def test_refuses_without_an_interactive_terminal(env):
    assert env.run(interactive=False) == 2
    assert env.asked == [] and env.assistant_calls == []


@pytest.mark.parametrize("prices", [{}, {"assistant-m": {"input_per_mtok": 3, "output_per_mtok": 15}}])
def test_refuses_to_run_without_prices(env, prices, capsys):
    env.set_prices(prices)

    assert env.run() == 2
    assert env.asked == [] and env.assistant_calls == []
    assert "unknown" in capsys.readouterr().out


def test_invalid_assistant_output_scores_zero_and_skips_the_judge(env):
    env.stdout = "Traceback: boom"

    env.run("--only", "c1")

    _, report = env.saved()
    result = report["cases"][0]
    assert result["case_score"] == 0 and not result["passed"] and result["judge"] is None
    assert "Traceback" in result["assistant"]["stdout_snippet"]
    assert env.client.calls == []


def test_wrong_fields_fail_the_case_even_with_a_high_judge_score(env):
    env.stdout = json.dumps({**GOOD_ANSWER, "coverage": "none"})

    env.run("--only", "c1")

    result = env.saved()[1]["cases"][0]
    assert not result["passed"]
    assert [c["name"] for c in result["checks"] if not c["passed"]] == ["coverage"]
    assert result["judge_score"] == 9


def test_only_runs_the_chosen_cases(env):
    env.run("--only", "b", "c1")

    assert env.assistant_calls == ["Question c1?", "Question c3?"]


def test_unknown_only_name_is_an_error(env):
    with pytest.raises(runner.loader.CaseError, match="nope"):
        env.run("--only", "nope")


def test_save_results_never_overwrites(tmp_path):
    runner.save_results(tmp_path, "s", {"a": 1}, WHEN)

    with pytest.raises(FileExistsError):
        runner.save_results(tmp_path, "s", {"a": 2}, WHEN)


def test_run_command_fills_placeholders_times_and_parses_tokens():
    script = "import sys; print(sys.argv[1]); print('Input tokens: 12', file=sys.stderr); print('Output tokens: 7', file=sys.stderr)"
    template = f"{{python}} -c \"{script}\" {{question}}"

    result = runner.run_command(template, "two words")

    assert result["stdout"].strip() == "two words"
    assert (result["input_tokens"], result["output_tokens"], result["returncode"]) == (12, 7, 0)
    assert result["seconds"] > 0 and result["error"] is None


def test_run_command_reports_a_timeout():
    result = runner.run_command('{python} -c "import time; time.sleep(5)"', "q", timeout=0.5)

    assert result["returncode"] is None and "timed out" in result["error"]
