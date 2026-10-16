"""Download, measure, QLoRA-train and activate Qwen on Apple Silicon."""

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from aac.config import ROOT
from aac.local_model import MODEL_REPO, parse_ids


def measure(model, tokenizer, rows, generate, sampler):
    hits, valid, results = 0, 0, []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        payload = json.loads(row["messages"][1]["content"])
        targets = json.loads(row["messages"][-1]["content"])["ids"]
        started = time.monotonic()
        output = generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=96, verbose=False)
        elapsed = round(time.monotonic() - started, 3)
        try:
            ids = parse_ids(output, payload["allowed"])
            valid += 1
            hit = bool(set(ids) & set(targets))
            hits += hit
        except ValueError:
            ids, hit = [], False
        results.append({"selected": payload["selected"], "target_ids": targets, "predicted_ids": ids,
                        "valid": bool(ids), "hit": hit, "seconds": elapsed, "raw_output": output})
    return {"examples": len(rows), "valid_output_rate": valid / len(rows), "hit_at_6": hits / len(rows), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    state = {"stage": "import_dependencies", "state": "running", "training_completed": False,
             "model_repo": MODEL_REPO, "iterations": args.iters}
    state_file = run_dir / "training-status.json"

    def status(**values):
        state.update(values)
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    try:
        status()
        from huggingface_hub import HfApi, snapshot_download
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        status(stage="download")
        print(f"Downloading {MODEL_REPO}…", flush=True)
        revision = HfApi().model_info(MODEL_REPO).sha
        model_path = Path(snapshot_download(MODEL_REPO, revision=revision,
                                           cache_dir=ROOT / "var/huggingface/hub",
                                           allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.md", "LICENSE*"]))
        status(model_revision=revision, model_path=str(model_path), stage="baseline")
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run_dir / "requirements-lock.txt").write_text(frozen.stdout)
        rows = [json.loads(line) for line in (run_dir / "data/test.jsonl").read_text().splitlines()]
        model, tokenizer = load(str(model_path), tokenizer_config={"trust_remote_code": False})
        max_tokens = 0
        for split in ("train", "valid", "test"):
            for line in (run_dir / f"data/{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                length = len(tokenizer.apply_chat_template(row["messages"], tokenize=True))
                max_tokens = max(max_tokens, length)
        if max_tokens > 2048:
            raise ValueError(f"A training example has {max_tokens} tokens; refusing silent truncation (limit 2048)")
        sampler = make_sampler(temp=0)
        before = measure(model, tokenizer, rows, generate, sampler)
        (run_dir / "before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n")
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
        adapter = run_dir / "adapter"
        command = [sys.executable, "-m", "mlx_lm.lora", "--model", str(model_path), "--train",
                   "--data", str(run_dir / "data"), "--adapter-path", str(adapter),
                   "--iters", str(args.iters), "--batch-size", "1", "--num-layers", "4",
                   "--learning-rate", "0.0001", "--max-seq-length", "2048", "--mask-prompt", "--grad-checkpoint",
                   "--steps-per-report", "10", "--steps-per-eval", "20", "--save-every", "20", "--val-batches", "-1"]
        status(stage="training", command=command, maximum_example_tokens=max_tokens)
        print(f"Training {args.iters} QLoRA iterations. Log: {run_dir / 'training.log'}", flush=True)
        with (run_dir / "training.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"))
        weights = adapter / "adapters.safetensors"
        if not weights.is_file() or not weights.stat().st_size:
            raise RuntimeError("Training exited without adapter weights")
        status(training_completed=True, stage="evaluate_adapter", adapter_path=str(adapter))
        model, tokenizer = load(str(model_path), adapter_path=str(adapter), tokenizer_config={"trust_remote_code": False})
        after = measure(model, tokenizer, rows, generate, sampler)
        (run_dir / "after.json").write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n")
        if after["valid_output_rate"] == 0:
            raise RuntimeError("Training completed but all validation generations were invalid; adapter not activated. Inspect training.log and after.json.")
        active = {"backend": "mlx", "model_repo": MODEL_REPO, "model_revision": revision, "model_path": str(model_path),
                  "adapter_path": str(adapter), "mlx_lm_version": importlib.metadata.version("mlx-lm"),
                  "trained_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir)}
        if not args.no_activate:
            temporary = ROOT / "var/active-model.json.tmp"
            temporary.write_text(json.dumps(active, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(ROOT / "var/active-model.json")
        status(state="completed", stage="evaluated" if args.no_activate else "activated")
        print(f"Before hit@6={before['hit_at_6']:.3f}; after hit@6={after['hit_at_6']:.3f}. Synthetic development examples only.", flush=True)
    except Exception as error:
        status(state="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
