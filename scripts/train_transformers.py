"""Train Qwen with CPU/CUDA LoRA or CUDA NF4 QLoRA; validate saved weights."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import subprocess
import sys

from aac.config import ROOT
from aac.torch_model import MODEL_REPO, encode_example, generate_text, load_model, resolve_device
from scripts.train_local import measure


def train_adapter(model, tokenizer, train_rows, valid_rows, adapter, iters, quantization, log):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quantization == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                        target_modules=["q_proj", "v_proj"],
                        layers_to_transform=list(range(max(0, model.config.num_hidden_layers - 4), model.config.num_hidden_layers)))
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_data = [encode_example(tokenizer, row) for row in train_rows]
    valid_data = [encode_example(tokenizer, row) for row in valid_rows]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0001)
    order = list(range(len(train_data)))
    rng = random.Random(42)

    def loss_for(example):
        tokens, labels = example
        inputs = torch.tensor([tokens], dtype=torch.long, device=model.device)
        targets = torch.tensor([labels], dtype=torch.long, device=model.device)
        return model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=targets).loss

    for step in range(iters):
        if step % len(order) == 0:
            rng.shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(train_data[order[step % len(order)]])
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss is not finite; no adapter activated")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == iters:
            print(f"Iteration {step + 1}/{iters}: train loss={loss.item():.5f}", file=log, flush=True)
        if (step + 1) % 20 == 0 or step + 1 == iters:
            model.eval()
            with torch.no_grad():
                valid_loss = sum(loss_for(row).item() for row in valid_data) / len(valid_data)
            print(f"Iteration {step + 1}: validation loss={valid_loss:.5f}", file=log, flush=True)
            model.save_pretrained(str(adapter), safe_serialization=True)
    # Release training references before the caller reloads the saved adapter.
    model.gradient_checkpointing_disable()
    model.config.use_cache = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--training", choices=("auto", "lora", "qlora"), default="auto")
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.iters <= 10000:
        parser.error("--iters must be between 1 and 10000")
    run_dir = args.run_dir.resolve()
    state = {"backend": "transformers", "stage": "import_dependencies", "state": "running",
             "training_completed": False, "model_repo": MODEL_REPO, "iterations": args.iters}

    def write(name, value):
        (run_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def status(**values):
        state.update(values)
        write("training-status.json", state)

    try:
        status()
        import torch
        from huggingface_hub import HfApi, snapshot_download
        torch.manual_seed(42)
        device = resolve_device(args.device)
        quantization = "4bit" if args.training == "qlora" or (args.training == "auto" and device == "cuda") else "none"
        if quantization == "4bit" and device != "cuda":
            raise ValueError("QLoRA on this backend requires CUDA. Use --training lora --device cpu.")
        if quantization == "4bit":
            import bitsandbytes  # noqa: F401 — fail before downloading if optional dependency is absent
        status(stage="download", device=device, quantization=quantization,
               training_method="qlora" if quantization == "4bit" else "lora")
        revision = HfApi().model_info(MODEL_REPO).sha
        print(f"Downloading {MODEL_REPO} for {device}…", flush=True)
        path = Path(snapshot_download(MODEL_REPO, revision=revision, cache_dir=ROOT / "var/huggingface/hub",
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "*.md"]))
        status(stage="baseline", model_revision=revision, model_path=str(path))
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (run_dir / "requirements-lock.txt").write_text(frozen, encoding="utf-8")
        datasets = {split: [json.loads(line) for line in (run_dir / f"data/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                    for split in ("train", "valid", "test")}
        model, tokenizer = load_model(path, device=device, quantization=quantization)
        for rows in datasets.values():
            for row in rows:
                encode_example(tokenizer, row)
        before = measure(model, tokenizer, datasets["test"], generate_text, None)
        write("before.json", before)
        adapter = run_dir / "adapter"
        status(stage="training")
        print(f"Training {args.iters} {state['training_method']} iterations on {device}. Log: {run_dir / 'training.log'}", flush=True)
        with (run_dir / "training.log").open("w", encoding="utf-8") as log:
            train_adapter(model, tokenizer, datasets["train"], datasets["valid"], adapter, args.iters, quantization, log)
        weights = adapter / "adapter_model.safetensors"
        if not weights.is_file() or not weights.stat().st_size:
            raise RuntimeError("Training exited without adapter weights")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        status(stage="evaluate_adapter", training_completed=True, adapter_path=str(adapter))
        model, tokenizer = load_model(path, adapter, device, quantization)
        after = measure(model, tokenizer, datasets["test"], generate_text, None)
        write("after.json", after)
        if after["valid_output_rate"] == 0:
            raise RuntimeError("Saved adapter produced no valid recommendations; active model unchanged. Inspect after.json.")
        active = {"backend": "transformers", "model_repo": MODEL_REPO, "model_revision": revision,
                  "model_path": str(path), "adapter_path": str(adapter), "quantization": quantization,
                  "trained_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir)}
        write("model.json", active)
        if not args.no_activate:
            temporary = ROOT / "var/active-model.json.tmp"
            temporary.write_text(json.dumps(active, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(ROOT / "var/active-model.json")
        status(state="completed", stage="evaluated" if args.no_activate else "activated")
        print(f"Before hit@6={before['hit_at_6']:.3f}; after hit@6={after['hit_at_6']:.3f}. Synthetic development data only.", flush=True)
    except Exception as error:
        status(state="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
