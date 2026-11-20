.PHONY: run test evaluate evaluate-agent benchmark-model training local-model check

run:
	python3 run.py

local-model:
	python3 setup_local.py --serve

test:
	python3 -m unittest discover -s tests -v

evaluate:
	python3 -m scripts.evaluate --output artifacts/evaluation.json

evaluate-agent:
	node scripts/evaluate-agent.mjs --output artifacts/agent-evaluation.json

benchmark-model:
	python3 -m scripts.benchmark_model --engines deterministic

training:
	python3 -m scripts.prepare_training data/scenarios.json --output artifacts/synthetic-training.jsonl

check: test
	python3 -m compileall -q aac scripts tests
