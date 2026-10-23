"""Run the warehouse eval with the same settings as `make eval`: python evals/run_warehouse.py [runner options].

Makes real API calls, so the runner still shows the estimated cost and waits for a "y".
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSISTANT_MODEL = "claude-sonnet-5-5"


def venv_python() -> Path:
    if os.name == "nt":
        return ROOT / "venv" / "Scripts" / "python.exe"
    return ROOT / "venv" / "bin" / "python"


def build_command(extra_args: list[str], assistant_model: str) -> list[str]:
    return [
        str(venv_python()),
        str(ROOT / "evals" / "runner.py"),
        "evals/warehouse",
        "--cmd", "{python} src/ask.py --json {question}",
        "--assistant-model", assistant_model,
        "--context-dir", "data/sops",
        *extra_args,
    ]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command = build_command(argv, os.environ.get("ASSISTANT_MODEL") or DEFAULT_ASSISTANT_MODEL)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
