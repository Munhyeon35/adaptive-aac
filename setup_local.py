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

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--data", type=Path, help="Reviewed export JSON; defaults to synthetic template examples")
    parser.add_argument("--prepare-only", action="store_true", help="Build data splits without downloading or training")
    parser.add_argument("--skip-install", action="store_true", help="Use the already installed .venv-mlx environment")
    parser.add_argument("--serve", action="store_true", help="Start AAC after successful training")
    args = parser.parse_args()
    if not 1 <= args.iters <= 10000:
        parser.error("--iters must be between 1 and 10000")
    from scripts.local_dataset import prepare_splits
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / "var/local-runs" / run_id
    run_dir.mkdir(parents=True)
    status_path = run_dir / "setup-status.json"
    state = {"run_id": run_id, "state": "running", "stage": "prepare_data", "requested_iterations": args.iters}

    def status(**values):
        state.update(values)
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    try:
        status()
        input_path = args.data
        if input_path is None:
            from scripts.synthetic_data import synthetic_records
            input_path = run_dir / "source-synthetic.json"
            input_path.write_text(json.dumps(synthetic_records(), ensure_ascii=False, indent=2) + "\n")
        manifest = prepare_splits(input_path, run_dir / "data")
        print(f"Prepared data: {manifest['example_counts']}\nRun directory: {run_dir}", flush=True)
        if args.prepare_only:
            status(state="prepared_only")
            return 0
        status(stage="check_platform")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This setup uses MLX on an Apple Silicon Mac (native arm64 Python).")
        env_path = ROOT / ".venv-mlx"
        python = env_path / "bin/python"
        if not python.exists():
            if args.skip_install:
                raise RuntimeError(".venv-mlx is missing; run without --skip-install first")
            status(stage="create_environment")
            venv.EnvBuilder(with_pip=True).create(env_path)
        env = dict(os.environ, HF_HOME=str(ROOT / "var/huggingface"), HF_HUB_DISABLE_TELEMETRY="1",
                   TOKENIZERS_PARALLELISM="false", PIP_DISABLE_PIP_VERSION_CHECK="1")
        if not args.skip_install:
            status(stage="install_dependencies")
            print("Installing local MLX training dependencies…", flush=True)
            subprocess.run([str(python), "-m", "pip", "install", "--retries", "0", "--timeout", "15",
                            "--cache-dir", str(ROOT / "var/pip-cache"), "-r", str(ROOT / "requirements-local.txt")],
                           cwd=ROOT, env=env, check=True)
        status(stage="download_train_evaluate")
        subprocess.run([str(python), "-m", "scripts.train_local", "--run-dir", str(run_dir),
                        "--iters", str(args.iters)], cwd=ROOT, env=env, check=True)
        status(state="completed", stage="activated")
        print("Local model training completed and activated. Start with: python3 " + str(ROOT / "run.py"), flush=True)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        status(state="failed", error=str(error))
        print(f"Stopped at {state['stage']}: {error}\nStatus: {status_path}\nNo new model was activated.", file=sys.stderr)
        return 1
    if args.serve:
        os.execv(str(python), [str(python), str(ROOT / "run.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
