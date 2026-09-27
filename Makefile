PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: install install-slim seed run test

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-laya.txt

install-slim:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

seed:
	$(PY) -m app.seed

run:
	$(PY) -m uvicorn app.main:app --reload

test:
	$(PY) -m pytest