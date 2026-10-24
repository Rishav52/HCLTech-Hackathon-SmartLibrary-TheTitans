.PHONY: setup seed run test eval calibrate
PY ?= python3

setup:            ## install core dependencies (add requirements-semantic.txt for local MiniLM embeddings)
	$(PY) -m pip install -r requirements.txt

seed:             ## (re)build the demo database
	$(PY) -m app.seed --reset

run:              ## start the app on http://localhost:8000 (auto-seeds on first start)
	$(PY) -m uvicorn app.main:app --port 8000

test:             ## unit + integration tests
	$(PY) -m pytest -q

eval:             ## evaluation report -> eval/out/report.md (exit 1 if a gate fails)
	$(PY) -m eval.run

calibrate:        ## measure the search abstention threshold for the active embedder
	$(PY) -m eval.calibrate
