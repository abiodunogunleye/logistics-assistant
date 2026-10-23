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
make eval      # run the warehouse eval (real API calls, costs money, manual only)
make run       # run src/main.py
make clean     # remove venv and caches
```

## Tests

Add tests under `tests/` for any new behavior. CI runs `pytest` automatically on every PR via `.github/workflows/ci.yml`.

## Evals

`evals/` holds a reusable eval pipeline (`runner.py` and its helper modules) and the warehouse suite (`evals/warehouse/cases.json` and `rubric.md`). The runner's own tests are in `tests/test_eval_*.py` and use fakes.

The eval itself makes real API calls (the assistant, then an Opus judge), so it never runs under `pytest` or in CI. `make eval` shows an estimated cost and waits for you to type `y`; it refuses to run in CI, without a terminal, or without prices for both models in `evals/pricing.json`. Use `make eval ARGS="--only safety_injury"` for a cheaper partial run. Each run saves a dated file in `evals/results/`, which is committed so runs can be compared.
