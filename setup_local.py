"""One command: prepare data, install MLX, download Qwen, train, evaluate, activate."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

from aac.runtime import select_backend, environment_dir, environment_python

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--data", type=Path, help="Reviewed export JSON; defaults to synthetic template examples")
    parser.add_argument("--prepare-only", action="store_true", help="Build data splits without downloading or training")
    parser.add_argument("--skip-install", action="store_true", help="Use the selected backend's existing environment")
    parser.add_argument("--serve", action="store_true", help="Start AAC after successful training")
    parser.add_argument("--backend", choices=("auto", "mlx", "transformers"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Transformers device")
    parser.add_argument("--training", choices=("auto", "lora", "qlora"), default="auto")
    parser.add_argument("--no-activate", action="store_true", help="Evaluate and save weights without replacing the active model")
    args = parser.parse_args()
    if not 1 <= args.iters <= 10000:
        parser.error("--iters must be between 1 and 10000")
    if args.no_activate and args.serve:
        parser.error("--serve cannot be combined with --no-activate")
    from scripts.local_dataset import prepare_splits
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "var/local-runs" / run_id
    run_dir.mkdir(parents=True)
    status_path = run_dir / "setup-status.json"
    state = {"run_id": run_id, "state": "running", "stage": "prepare_data", "requested_iterations": args.iters}

    def status(**values):
        state.update(values)
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        status()
        input_path = args.data
        if input_path is None:
            from scripts.synthetic_data import synthetic_records
            input_path = run_dir / "source-synthetic.json"
            input_path.write_text(json.dumps(synthetic_records(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = prepare_splits(input_path, run_dir / "data")
        print(f"Prepared data: {manifest['example_counts']}\nRun directory: {run_dir}", flush=True)
        if args.prepare_only:
            status(state="prepared_only")
            return 0
        status(stage="check_platform")
        backend = select_backend(args.backend)
        if backend == "mlx" and (args.device != "auto" or args.training == "lora"):
            raise ValueError("The MLX backend uses Apple GPU QLoRA. Use --backend transformers for CPU/CUDA LoRA.")
        if backend == "transformers" and platform.system() == "Darwin" and platform.machine() == "x86_64" and sys.version_info >= (3, 13):
            raise RuntimeError("Intel Mac requires Python 3.11 or 3.12 for the PyTorch 2.2.2 wheels.")
        status(backend=backend)
        print(f"Local backend: {backend}", flush=True)
        env_path = environment_dir(backend)
        python = environment_python(env_path)
        if not python.exists():
            if args.skip_install:
                raise RuntimeError(f"{env_path.name} is missing; run without --skip-install first")
            status(stage="create_environment")
            venv.EnvBuilder(with_pip=True).create(env_path)
        env = dict(os.environ, HF_HOME=str(ROOT / "var/huggingface"), HF_HUB_DISABLE_TELEMETRY="1",
                   TOKENIZERS_PARALLELISM="false", PIP_DISABLE_PIP_VERSION_CHECK="1",
                   PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        if not args.skip_install:
            status(stage="install_dependencies")
            print(f"Installing local {backend} training dependencies…", flush=True)
            requirements = "requirements-local.txt" if backend == "mlx" else "requirements-torch.txt"
            subprocess.run([str(python), "-m", "pip", "install", "--retries", "2", "--timeout", "30",
                            "--cache-dir", str(ROOT / "var/pip-cache"), "-r", str(ROOT / requirements)],
                           cwd=ROOT, env=env, check=True)
            if backend == "transformers" and args.training != "lora":
                device = subprocess.check_output([str(python), "-c", "from aac.torch_model import resolve_device; "
                                                 f"print(resolve_device({args.device!r}))"], cwd=ROOT, env=env, text=True).strip()
                if device == "cuda":
                    subprocess.run([str(python), "-m", "pip", "install", "--cache-dir", str(ROOT / "var/pip-cache"),
                                    "bitsandbytes>=0.45,<2"], cwd=ROOT, env=env, check=True)
        status(stage="download_train_evaluate")
        module = "scripts.train_local" if backend == "mlx" else "scripts.train_transformers"
        command = [str(python), "-m", module, "--run-dir", str(run_dir), "--iters", str(args.iters)]
        if backend == "transformers":
            command += ["--device", args.device, "--training", args.training]
        if args.no_activate:
            command.append("--no-activate")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        status(state="completed", stage="evaluated" if args.no_activate else "activated")
        print("Local model training completed." + (" Active model unchanged." if args.no_activate else
              " Start with: python " + str(ROOT / "run.py")), flush=True)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        status(state="failed", error=str(error))
        print(f"Stopped at {state['stage']}: {error}\nStatus: {status_path}\nNo new model was activated.", file=sys.stderr)
        return 1
    if args.serve:
        os.execv(str(python), [str(python), str(ROOT / "run.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
