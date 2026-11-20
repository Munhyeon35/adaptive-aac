"""Local inference in a bounded worker, using portable pipe I/O on every OS."""

import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from .config import ROOT
from .runtime import active_backend
from .vocabulary import BY_ID

MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def local_instructions(limit=6):
    # The limit=6 text is byte-identical to the prompt existing adapters were trained on.
    return ('Predict the next AAC symbols. Return only JSON: {"ids":["id",...]}. '
            f"Choose 1 to {limit} unique IDs from allowed. Use the user's selections, context, and history. "
            "Preserve negation. Never choose an already selected ID. The user makes the final choice.")


LOCAL_INSTRUCTIONS = local_instructions(6)


def output_token_budget(limit):
    # Mulberry IDs ("mulberry:Advent_calendar") need ~10 tokens each; 96 covers six base IDs.
    return min(256, max(96, 16 + 10 * limit))


def local_messages(selected, history, candidates, context, limit=6):
    payload = {"selected": selected, "context": context, "limit": limit,
               "history": [row["symbols"] for row in history[:6]],
               "allowed": {i: BY_ID[i]["label"] for i in sorted(candidates)}}
    return [{"role": "system", "content": local_instructions(limit)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def parse_ids(text, candidates, limit=6):
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError("Invalid local model output")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    result = json.loads(text)
    ids = result.get("ids") if isinstance(result, dict) else None
    if (not isinstance(ids, list) or not 1 <= len(ids) <= limit
            or any(not isinstance(i, str) or i not in candidates for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Local model returned invalid IDs")
    return ids


class WorkerGenerationError(ValueError):
    """One request failed inside a healthy worker; the loaded model is kept."""


WORKER_LOG = ROOT / "var/logs/local-worker.log"
WORKER_LOG_LIMIT = 1_000_000


class LocalModelProvider:
    source = "local_llm"

    def __init__(self, model_path, adapter_path=None, timeout=None, backend="mlx", device="auto", quantization="none",
                 load_timeout=None):
        if backend not in {"mlx", "transformers"}:
            raise ValueError("Unknown local inference backend")
        if device not in {"auto", "cpu", "cuda"} or quantization not in {"none", "4bit"}:
            raise ValueError("Invalid local model device or quantization")
        self.backend, self.device, self.quantization = backend, device, quantization
        self.model_path = Path(model_path).resolve()
        self.adapter_path = Path(adapter_path).resolve() if adapter_path else None
        if not (self.model_path / "config.json").is_file():
            raise ValueError("Local model is not downloaded. Run setup_local.py first.")
        weights = "adapters.safetensors" if backend == "mlx" else "adapter_model.safetensors"
        if self.adapter_path and not (self.adapter_path / weights).is_file():
            raise ValueError("Local adapter file is missing.")
        # Generation and model loading have separate budgets: a cold CPU load must not
        # be mistaken for a hung generation (which previously killed and reloaded the model).
        self.timeout = timeout if timeout is not None else (25 if backend == "mlx" else 120)
        self.load_timeout = load_timeout if load_timeout is not None else 300
        self._process = None
        self._responses = None
        self._log = None
        self._lock = threading.Lock()
        self._retry_at = 0
        atexit.register(self.close)

    def _open_log(self):
        # Diagnostics only: library warnings and exception class names. Workers never print
        # prompts, selections, or history, so no communication content reaches this file.
        try:
            WORKER_LOG.parent.mkdir(parents=True, exist_ok=True)
            mode = "wb" if WORKER_LOG.exists() and WORKER_LOG.stat().st_size > WORKER_LOG_LIMIT else "ab"
            return open(WORKER_LOG, mode)
        except OSError:
            return None

    def _start(self):
        if self._process and self._process.poll() is None:
            return
        module = "aac.mlx_worker" if self.backend == "mlx" else "aac.transformers_worker"
        command = [sys.executable, "-u", "-m", module, "--model", str(self.model_path)]
        if self.backend == "transformers":
            command += ["--device", self.device, "--quantization", self.quantization]
        if self.adapter_path:
            command.extend(["--adapter", str(self.adapter_path)])
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HOME=str(ROOT / "var/huggingface"))
        self._log = self._open_log()
        self._process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self._log or subprocess.DEVNULL, env=env)
        self._responses = queue.Queue(maxsize=1)
        threading.Thread(target=self._read_output, args=(self._process.stdout, self._responses), daemon=True).start()
        try:
            ready = self._read_response(self.load_timeout, "Local model load timed out")
        except (OSError, ValueError, TypeError):
            self.close()
            raise
        if ready.get("ready") is not True:
            self.close()
            raise OSError("Local model failed to load")

    def _read_response(self, timeout, timeout_message):
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(timeout_message) from None
        if not line or len(line) > 16000:
            raise ValueError("Local worker closed or returned an oversized response")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("Invalid local worker response")
        return response

    def preload(self):
        """Load the worker in the background so the first recommendation is not a cold start.

        Requests arriving meanwhile find the lock busy and use the deterministic board.
        """
        def load():
            if not self._lock.acquire(timeout=1):
                return
            try:
                self._start()
            except (OSError, ValueError, TypeError):
                self._retry_at = time.monotonic() + 10
            finally:
                self._lock.release()
        threading.Thread(target=load, daemon=True).start()

    @staticmethod
    def _read_output(stream, responses):
        # select() cannot read subprocess pipes on Windows. One reader belongs to
        # each worker; killing that worker unblocks it without contaminating a restart.
        try:
            while True:
                line = stream.readline(16001)
                responses.put_nowait(line)
                if not line or len(line) > 16000:
                    return
        except (OSError, ValueError, queue.Full):
            return

    def close(self):
        process, self._process = self._process, None
        if process:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
        log, self._log = self._log, None
        if log:
            log.close()
        self._responses = None

    def _generate(self, messages, max_tokens=96):
        if time.monotonic() < self._retry_at:
            raise OSError("Local model is cooling down after an error")
        if not self._lock.acquire(timeout=1):
            raise TimeoutError("Local model busy")
        try:
            self._start()
            self._process.stdin.write((json.dumps({"messages": messages, "max_tokens": max_tokens}, ensure_ascii=False) + "\n").encode())
            self._process.stdin.flush()
            response = self._read_response(self.timeout, "Local generation timed out")
            if response.get("error") == "generation_failed":
                raise WorkerGenerationError("Local generation failed")
            if not isinstance(response.get("text"), str):
                raise ValueError("Invalid local generation response")
            return response["text"]
        except WorkerGenerationError:
            raise  # The worker answered, so its pipe and loaded model are still usable.
        except (OSError, ValueError, TypeError):
            self.close()
            self._retry_at = time.monotonic() + 10
            raise
        finally:
            self._lock.release()

    def recommend(self, selected, history, candidates, context, limit):
        text = self._generate(local_messages(selected, history, candidates, context, limit), output_token_budget(limit))
        return parse_ids(text, candidates, limit)

    # Training targets next-symbol prediction; deterministic realization preserves meaning.
    def sentence(self, selected, options):
        return options[0]


class LocalMLXProvider(LocalModelProvider):
    """Backward-compatible name for existing MLX callers."""


def configured_local_provider():
    mode = os.environ.get("AAC_LOCAL_MODEL", "auto")
    if mode == "off":
        return None
    manifest = ROOT / "var/active-model.json"
    if mode == "auto":
        if not manifest.exists():
            return None
        config = json.loads(manifest.read_text(encoding="utf-8"))
        return LocalModelProvider(config["model_path"], config.get("adapter_path"), backend=active_backend(),
                                  device=os.environ.get("AAC_LOCAL_DEVICE", "auto"),
                                  quantization=config.get("quantization", "none"))
    path = Path(mode)
    if not path.is_absolute():
        path = ROOT / path
    adapter = os.environ.get("AAC_LOCAL_ADAPTER")
    if adapter and not Path(adapter).is_absolute():
        adapter = str(ROOT / adapter)
    return LocalModelProvider(path, adapter, backend=active_backend(), device=os.environ.get("AAC_LOCAL_DEVICE", "auto"))
