# Contributing

## Setup

```
make install
cp .env.example .env   # then add your ANTHROPIC_API_KEY
./venv/Scripts/pre-commit install   # Windows; use venv/bin/pre-commit on macOS/Linux
```

This installs a git hook that blocks commits containing `.env`-style files (except `.env.example`).

## Workflow

- `master` is protected: changes must go through a pull request, and the `test` check must pass before merging.
- Create a branch, commit your changes, and open a PR against `master`.
- Direct pushes to `master` are blocked, including for repo admins.

## Common commands

```
make install   # create venv and install dependencies
make test      # run the test suite
make run       # run src/main.py
make clean     # remove venv and caches
```

## Tests

Add tests under `tests/` for any new behavior. CI runs `pytest` automatically on every PR via `.github/workflows/ci.yml`.
