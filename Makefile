.PHONY: run test evaluate training local-model check

run:
	python3 run.py

local-model:
	python3 setup_local.py --serve

test:
	python3 -m unittest discover -s tests -v

evaluate:
	python3 -m scripts.evaluate --output artifacts/evaluation.json

training:
	python3 -m scripts.prepare_training data/scenarios.json --output artifacts/synthetic-training.jsonl

check: test
	python3 -m compileall -q aac scripts tests
