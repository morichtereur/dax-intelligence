.PHONY: install ingest run test eval-retrieval eval-grounding

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:  ## Create the venv and install runtime + dev dependencies
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt -r requirements-dev.txt

ingest:  ## Chunk and index every PDF in data/raw/
	$(PY) pipeline/ingest.py

run:  ## Launch the Streamlit app
	$(VENV)/bin/streamlit run app/app.py

test:  ## Unit tests — synthetic fixtures, no API key or real reports needed
	$(VENV)/bin/pytest tests/ -q

eval-retrieval:  ## precision@k / recall@k against eval/gold_queries.json (free, local)
	$(PY) eval/eval_retrieval.py

eval-grounding:  ## Citation grounding + faithfulness judge — COSTS REAL API CALLS
	$(PY) eval/eval_grounding.py
