import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check-no-env.sh"


def _bash_executable():
    if sys.platform == "win32":
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(candidate).exists():
                return candidate
    found = shutil.which("bash")
    if found:
        return found
    raise RuntimeError("bash executable not found")


BASH = _bash_executable()


@pytest.fixture
def git_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    return tmp_path


def run_hook(repo):
    return subprocess.run(
        [BASH, str(SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_blocks_staged_env_file(git_repo):
    (git_repo / ".env").write_text("ANTHROPIC_API_KEY=secret\n")
    subprocess.run(["git", "add", "-f", ".env"], cwd=git_repo, check=True)

    result = run_hook(git_repo)

    assert result.returncode != 0
    assert ".env" in result.stdout


def test_allows_env_example(git_repo):
    (git_repo / ".env.example").write_text("ANTHROPIC_API_KEY=\n")
    subprocess.run(["git", "add", ".env.example"], cwd=git_repo, check=True)

    result = run_hook(git_repo)

    assert result.returncode == 0


def test_allows_unrelated_file(git_repo):
    (git_repo / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "README.md"], cwd=git_repo, check=True)

    result = run_hook(git_repo)

    assert result.returncode == 0
