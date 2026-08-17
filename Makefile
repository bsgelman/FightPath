PYTHON := .venv/Scripts/python
PIP    := .venv/Scripts/pip

.PHONY: install ingest features train backtest predict test clean

install:
	python -m venv .venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

ingest:
	$(PYTHON) scripts/01_ingest.py

features:
	$(PYTHON) scripts/02_build_features.py

train:
	$(PYTHON) scripts/03_train.py

backtest:
	$(PYTHON) scripts/04_backtest.py
	$(PYTHON) scripts/05_evaluate_props.py

predict:
	$(PYTHON) predict.py $(ARGS)

test:
	.venv/Scripts/pytest tests/ -v

clean:
	rm -rf data/interim/*.parquet data/processed/*.parquet outputs/models/*.joblib
