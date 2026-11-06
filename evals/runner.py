"""Generic eval runner: load cases, run the assistant, grade, score, save. Nothing here is project-specific.

Manual use only. It makes real API calls, so it refuses to run in CI or without a terminal, shows the
estimated cost, and waits for a "y". Run `python evals/runner.py --help` or see the `eval` target in the Makefile.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import costing
import graders
import judge
import loader
import scoring

HERE = Path(__file__).parent
TIMEOUT_SECONDS = 300
SNIPPET_CHARS = 500
MAX_CONSECUTIVE_FAILURES = 3
INPUT_TOKENS = re.compile(r"Input tokens:\s*(\d+)")
OUTPUT_TOKENS = re.compile(r"Output tokens:\s*(\d+)")


def run_command(template: str, question: str, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Run the assistant once. {python} and {question} in the template are filled in after it is split."""
    args = [part.replace("{python}", sys.executable).replace("{question}", question) for part in shlex.split(template)]
    started = time.perf_counter()
    try:
        done = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "returncode": None, "seconds": time.perf_counter() - started,
                "input_tokens": 0, "output_tokens": 0, "error": f"timed out after {timeout}s"}

    def tokens(pattern):
        match = pattern.search(done.stderr)
        return int(match.group(1)) if match else 0

    return {"stdout": done.stdout, "stderr": done.stderr, "returncode": done.returncode,
            "seconds": time.perf_counter() - started, "input_tokens": tokens(INPUT_TOKENS),
            "output_tokens": tokens(OUTPUT_TOKENS), "error": None}


def evaluate_case(case: dict, run_assistant, client, judge_model: str, system: list[dict]) -> dict:
    run = run_assistant(case["question"])
    answer = graders.parse_answer(run["stdout"])
    checks = graders.grade_answer(case, answer)
    judgement = judge.judge_case(client, judge_model, system, case, answer) if answer is not None else None

    return {
        "id": case["id"], "category": case["category"], "question": case["question"],
        **scoring.score_case(checks, judgement),
        "checks": checks,
        "judge": asdict(judgement) if judgement else None,
        "assistant": {
            "answer": answer, "returncode": run["returncode"], "error": run["error"],
            "stdout_snippet": None if answer else run["stdout"][:SNIPPET_CHARS],
            "stderr_snippet": None if answer else run["stderr"][-SNIPPET_CHARS:],
        },
        "seconds": round(run["seconds"], 2),
        "assistant_input_tokens": run["input_tokens"], "assistant_output_tokens": run["output_tokens"],
        "judge_input_tokens": judgement.input_tokens if judgement else 0,
        "judge_output_tokens": judgement.output_tokens if judgement else 0,
    }


def call_failed(result: dict) -> bool:
    """The assistant command itself failed (non-zero exit or timeout), as opposed to giving a bad answer."""
    return bool(result["assistant"]["error"]) or result["assistant"]["returncode"] != 0


def last_error_line(result: dict) -> str:
    lines = (result["assistant"]["stderr_snippet"] or "").strip().splitlines()
    return lines[-1] if lines else (result["assistant"]["error"] or "no error output")


def git_sha() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return done.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_results(results_dir: Path, suite: str, report: dict, when: datetime) -> Path:
    """Write a new dated file; never overwrite an earlier run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{when:%Y-%m-%dT%H%M%S}_{suite}.json"
    with open(path, "x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return path


def print_summary(report: dict, actual_cost: str) -> None:
    print(f"\n{'case':<16}{'pass':<6}{'score':>6}{'code':>6}{'judge':>6}{'secs':>7}{'tok in/out':>14}")
    for case in report["cases"]:
        tokens = f"{case['assistant_input_tokens']}/{case['assistant_output_tokens']}"
        judge_score = "-" if case["judge_score"] is None else case["judge_score"]
        print(f"{case['id']:<16}{'yes' if case['passed'] else 'NO':<6}{case['case_score']:>6}{case['code_score']:>6}"
              f"{judge_score:>6}{case['seconds']:>7}{tokens:>14}")
    print("\nBy category (passed/cases, mean score):")
    for name, stats in report["summary"]["by_category"].items():
        print(f"  {name:<22}{stats['passed']}/{stats['cases']}  {stats['mean_score']}")
    overall = report["summary"]["overall"]
    print(f"\nOverall: {overall['passed']}/{overall['cases']} passed, mean score {overall['mean_score']}, "
          f"{overall['total_seconds']}s (includes Python start-up per case)")
    print(f"Tokens: assistant {overall['total_assistant_input_tokens']}/{overall['total_assistant_output_tokens']}, "
          f"judge {overall['total_judge_input_tokens']}/{overall['total_judge_output_tokens']}. {actual_cost}")


def default_client():
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv(HERE.parent / ".env")
    return Anthropic()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run an eval suite against the assistant (real API calls).")
    parser.add_argument("suite_dir", type=Path, help="folder with cases.json and rubric.md")
    parser.add_argument("--cmd", required=True, help="assistant command; {python} and {question} are filled in")
    parser.add_argument("--assistant-model", required=True, help="model the assistant uses (for the cost estimate)")
    parser.add_argument("--judge-model", default=judge.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--context-dir", type=Path, help="folder of .md files (the SOPs) given to the judge")
    parser.add_argument("--only", nargs="*", default=[], help="case ids or categories to run")
    parser.add_argument("--pricing", type=Path, default=HERE / "pricing.json")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def main(argv=None, *, run_assistant=None, client_factory=default_client, input_fn=input,
         now=datetime.now, is_interactive=lambda: sys.stdin.isatty()) -> int:
    args = parse_args(argv)
    if os.environ.get("CI"):
        print("Refusing to run: the eval makes real API calls and must not run in CI.", file=sys.stderr)
        return 2
    if not is_interactive():
        print("Refusing to run without an interactive terminal: it needs you to confirm the cost.", file=sys.stderr)
        return 2

    suite = args.suite_dir.name
    cases = loader.select_cases(loader.load_cases(args.suite_dir / "cases.json"), args.only)
    rubric = (args.suite_dir / "rubric.md").read_text(encoding="utf-8")
    context = judge.load_context(args.context_dir) if args.context_dir else ""
    pricing = costing.load_pricing(args.pricing)

    estimate = costing.estimate_run(len(cases), len(context), len(rubric), pricing, args.assistant_model, args.judge_model)
    print(costing.format_estimate(estimate, args.assistant_model, args.judge_model))
    if estimate["usd"]["total"] is None:
        print(f"Add prices for both models to {args.pricing} first, so the cost is known before spending.", file=sys.stderr)
        return 2
    if input_fn("Run? [y/N] ").strip().lower() != "y":
        print("Aborted. No API calls were made.")
        return 1

    run_assistant = run_assistant or partial(run_command, args.cmd, timeout=args.timeout)
    client, system, started = client_factory(), judge.build_system(rubric, context), now()
    results, failed_in_a_row = [], 0
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']} ...", flush=True)
        results.append(evaluate_case(case, run_assistant, client, args.judge_model, system))
        failed_in_a_row = failed_in_a_row + 1 if call_failed(results[-1]) else 0
        if failed_in_a_row >= MAX_CONSECUTIVE_FAILURES:
            print(f"\nStopped after {failed_in_a_row} failed assistant calls in a row; nothing saved. "
                  f"Last error: {last_error_line(results[-1])}", file=sys.stderr)
            return 3

    summary = scoring.summarize(results)
    total = summary["overall"]
    actual = [costing.cost_usd(pricing, model, total[f"total_{who}_input_tokens"], total[f"total_{who}_output_tokens"])
              for who, model in (("assistant", args.assistant_model), ("judge", args.judge_model))]
    actual_cost = f"Cost (upper bound): {costing.format_usd(sum(actual))}" if None not in actual else "Cost: unknown"
    report = {
        "suite": suite, "started_at": started.isoformat(timespec="seconds"),
        "config": {
            "assistant_model": args.assistant_model, "judge_model": args.judge_model, "cmd": args.cmd,
            "claude_env": {k: v for k, v in os.environ.items() if k.startswith("CLAUDE_")}, "git_sha": git_sha(),
        },
        "estimate": estimate, "summary": summary, "cases": results,
    }
    path = save_results(args.results_dir, suite, report, started)
    print_summary(report, actual_cost)
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
