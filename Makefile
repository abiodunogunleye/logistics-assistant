VENV := venv

ifeq ($(OS),Windows_NT)
	PYTHON := $(VENV)/Scripts/python.exe
else
	PYTHON := $(VENV)/bin/python
endif

ASSISTANT_MODEL ?= claude-sonnet-5-5

.PHONY: venv install test eval run clean

venv:
	python -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

# Makes real API calls and costs money: never part of `make test` or CI. Pass extra options with ARGS="--only safety_injury".
eval:
	$(PYTHON) evals/runner.py evals/warehouse --cmd "{python} src/ask.py --json {question}" --assistant-model $(ASSISTANT_MODEL) --context-dir data/sops $(ARGS)

run:
	$(PYTHON) src/main.py

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
