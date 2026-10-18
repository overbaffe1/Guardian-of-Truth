.PHONY: install test devset predict heuristic

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

devset:
	python scripts/make_dev_from_tau2.py --tau2-root /tmp/tau2 --out data/dev_synthetic.jsonl --per-domain 40

predict:
	python scripts/predict.py --input test.parquet --output predictions.parquet

heuristic:
	python scripts/predict.py --input test.parquet --output predictions.parquet --backend heuristic
