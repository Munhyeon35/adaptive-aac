"""Local MLX inference in a bounded, persistent worker process. No HTTP service needed."""

import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from .config import ROOT
from .vocabulary import BY_ID

MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LOCAL_INSTRUCTIONS = (
    'Predict the next AAC symbols. Return only JSON: {"ids":["id",...]}. '
    "Choose 1 to 6 unique IDs from allowed. Use the user's selections, context, and history. "
    "Preserve negation. Never choose an already selected ID. The user makes the final choice."
)


def local_messages(selected, history, candidates, context, limit=6):
    payload = {"selected": selected, "context": context, "limit": limit,
               "history": [row["symbols"] for row in history[:6]],
               "allowed": {i: BY_ID[i]["label"] for i in sorted(candidates)}}
    return [{"role": "system", "content": LOCAL_INSTRUCTIONS},
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


class LocalMLXProvider:
    source = "local_llm"

    def __init__(self, model_path, adapter_path=None, timeout=25):
        self.model_path = Path(model_path).resolve()
        self.adapter_path = Path(adapter_path).resolve() if adapter_path else None
        if not (self.model_path / "config.json").is_file():
            raise ValueError("Local model is not downloaded. Run setup_local.py first.")
        if self.adapter_path and not (self.adapter_path / "adapters.safetensors").is_file():
            raise ValueError("Local adapter file is missing.")
        self.timeout = timeout
        self._process = None
        self._lock = threading.Lock()
        self._retry_at = 0
        atexit.register(self.close)

    def _start(self):
        if self._process and self._process.poll() is None:
            return
        command = [sys.executable, "-u", "-m", "aac.mlx_worker", "--model", str(self.model_path)]
        if self.adapter_path:
            command.extend(["--adapter", str(self.adapter_path)])
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        self._process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=env)

    def close(self):
        process, self._process = self._process, None
        if process:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()

    def _generate(self, messages):
        if time.monotonic() < self._retry_at:
            raise OSError("Local model is cooling down after an error")
        if not self._lock.acquire(timeout=1):
            raise TimeoutError("Local model busy")
        try:
            self._start()
            self._process.stdin.write((json.dumps({"messages": messages}, ensure_ascii=False) + "\n").encode())
            self._process.stdin.flush()
            if not select.select([self._process.stdout], [], [], self.timeout)[0]:
                raise TimeoutError("Local generation timed out")
            response = json.loads(self._process.stdout.readline(16001))
            if not isinstance(response, dict) or "text" not in response:
                raise ValueError("Local generation failed")
            return response["text"]
        except (OSError, ValueError, TypeError):
            self.close()
            self._retry_at = time.monotonic() + 10
            raise
        finally:
            self._lock.release()

    def recommend(self, selected, history, candidates, context, limit):
        return parse_ids(self._generate(local_messages(selected, history, candidates, context, limit)), candidates, limit)

    # Training targets next-symbol prediction; deterministic realization preserves meaning.
    def sentence(self, selected, options):
        return options[0]


def configured_local_provider():
    mode = os.environ.get("AAC_LOCAL_MODEL", "auto")
    if mode == "off":
        return None
    manifest = ROOT / "var/active-model.json"
    if mode == "auto":
        if not manifest.exists():
            return None
        config = json.loads(manifest.read_text())
        return LocalMLXProvider(config["model_path"], config.get("adapter_path"))
    path = Path(mode)
    if not path.is_absolute():
        path = ROOT / path
    adapter = os.environ.get("AAC_LOCAL_ADAPTER")
    if adapter and not Path(adapter).is_absolute():
        adapter = str(ROOT / adapter)
    return LocalMLXProvider(path, adapter)
