# Logistics Assistant

An AI-powered warehouse assistant built on the Claude API, Model Context Protocol (MCP) tools, and the Claude Agent SDK.

The assistant will help with warehouse operations tasks such as inventory lookups, shipment tracking, and logistics workflow automation by combining Claude's reasoning with tool-calling access to warehouse systems and data.

## Project Structure

- `src/` — application source code
- `tests/` — test suite
- `data/` — sample data and fixtures
- `docs/` — documentation

## Status

[![CI](https://github.com/abiodunogunleye/logistics-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/abiodunogunleye/logistics-assistant/actions/workflows/ci.yml)

## Setup

1. Copy `.env.example` to `.env` and add your Anthropic API key.
2. (Setup instructions for dependencies and running the project will be added as the project develops.)

## Web app (demo)

A Streamlit chat app (`streamlit_app.py`) answers questions from the SOPs with streaming text, the cited SOPs, an escalation badge (FLM / Senior FLM / Shift Manager) and a red banner when the answer flags a safety alert. It uses the same system prompt, SOPs and settings (`src/chat.py`, `src/config.py`) as the command-line tools. The sidebar shows the score of the newest full eval run saved in `evals/results/`, meaning one that ran every case in `cases.json` (`--only` runs and regrades are skipped); the app never runs an eval.

Run it locally (needs `ANTHROPIC_API_KEY` in `.env`):

```
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

On macOS or Linux: `venv/bin/python -m streamlit run streamlit_app.py`.

### Cost limits

- 10 questions per visitor session, and 200 questions a day across all visitors (UTC). Change the daily cap with `DAILY_QUESTION_CAP`.
- Questions are limited to 500 characters.
- Both limits live in memory: reloading the page resets a visitor's count, and the daily count resets if the app restarts or sleeps. The real backstop is a monthly spend limit on the API key in the Anthropic console, so set one.

### Deploy on Streamlit Community Cloud

1. Create an app from this repo with `streamlit_app.py` as the main file.
2. In the app's Secrets box, add `ANTHROPIC_API_KEY` (and optionally `DAILY_QUESTION_CAP`). See `.streamlit/secrets.toml.example`. The key is only ever read from Streamlit secrets or `.env`; both are git-ignored and blocked by the pre-commit hook.
