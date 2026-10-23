import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

import run_warehouse  # noqa: E402


def test_build_command_matches_make_eval():
    command = run_warehouse.build_command([], "claude-sonnet-5-5")
    assert Path(command[0]).parts[-3] == "venv"
    assert Path(command[1]) == run_warehouse.ROOT / "evals" / "runner.py"
    assert command[2:] == [
        "evals/warehouse",
        "--cmd", "{python} src/ask.py --json {question}",
        "--assistant-model", "claude-sonnet-5-5",
        "--context-dir", "data/sops",
    ]


def test_build_command_passes_extra_args_and_model():
    command = run_warehouse.build_command(["--only", "safety_injury"], "other-model")
    assert command[-2:] == ["--only", "safety_injury"]
    assert command[command.index("--assistant-model") + 1] == "other-model"


def test_main_runs_from_project_root_and_returns_exit_code(monkeypatch):
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))
        return type("Done", (), {"returncode": 2})()

    monkeypatch.setattr(run_warehouse.subprocess, "run", fake_run)
    monkeypatch.delenv("ASSISTANT_MODEL", raising=False)
    assert run_warehouse.main(["--only", "x"]) == 2
    command, cwd = calls[0]
    assert cwd == run_warehouse.ROOT
    assert command[command.index("--assistant-model") + 1] == run_warehouse.DEFAULT_ASSISTANT_MODEL
    assert command[-2:] == ["--only", "x"]
