"""Runtime selection without importing either optional ML framework."""

import json
import os
from pathlib import Path
import platform

from .config import ROOT


def select_backend(requested="auto", system=None, machine=None):
    system = system or platform.system()
    machine = machine or platform.machine()
    if requested not in {"auto", "mlx", "transformers"}:
        raise ValueError("Backend must be auto, mlx, or transformers")
    apple_silicon = system == "Darwin" and machine.lower() == "arm64"
    if requested == "mlx" and not apple_silicon:
        raise ValueError("MLX requires Apple Silicon. Use --backend transformers on this computer.")
    return ("mlx" if apple_silicon else "transformers") if requested == "auto" else requested


def environment_dir(backend):
    return ROOT / (".venv-mlx" if backend == "mlx" else ".venv-torch")


def environment_python(directory, system=None):
    return Path(directory) / ("Scripts/python.exe" if (system or platform.system()) == "Windows" else "bin/python")


def active_backend():
    mode = os.environ.get("AAC_LOCAL_MODEL", "auto")
    manifest = ROOT / "var/active-model.json"
    if mode == "auto" and manifest.exists():
        config = json.loads(manifest.read_text(encoding="utf-8"))
        # Manifests from the original implementation are MLX models.
        return select_backend(config.get("backend", "mlx"))
    return select_backend(os.environ.get("AAC_LOCAL_BACKEND", "auto"))
