PY ?= python
export PYTHONPATH := src

.PHONY: all ingest transform train test clean
all:
	$(PY) -m smbcredit.pipeline
ingest:
	$(PY) -m smbcredit.ingest
transform: ingest
	$(PY) -m smbcredit.transform
test:
	$(PY) -m pytest -q tests
clean:
	rm -rf data/processed/* reports/figures/*.png reports/*.csv reports/metrics.json
