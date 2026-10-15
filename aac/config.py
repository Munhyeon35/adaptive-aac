import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=None):
    path = Path(path or ROOT / ".env")
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # .env is data, never executed as shell code.
                if key.strip() in {"OPENAI_API_KEY", "OPENAI_MODEL", "AAC_PORT", "AAC_DB_PATH"}:
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
