"""Start AAC from any working directory, using the local MLX venv if installed."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    venv = ROOT / ".venv-mlx"
    interpreter = venv / "bin/python"
    if interpreter.exists() and Path(sys.prefix) != venv:
        os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]])
    from aac.server import main
    main()
