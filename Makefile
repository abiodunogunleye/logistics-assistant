VENV := venv

ifeq ($(OS),Windows_NT)
	PYTHON := $(VENV)/Scripts/python.exe
else
	PYTHON := $(VENV)/bin/python
endif

.PHONY: venv install test run clean

venv:
	python -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) src/main.py

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
