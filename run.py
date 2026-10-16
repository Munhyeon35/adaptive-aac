"""Start AAC using the active model's environment on Windows, macOS, or Linux."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    from aac.config import load_env
    from aac.runtime import active_backend, environment_dir, environment_python
    load_env()
    try:
        venv = environment_dir(active_backend())
    except (ValueError, KeyError) as error:
        raise SystemExit(f"Local model configuration error: {error}") from None
    interpreter = environment_python(venv)
    if interpreter.exists() and Path(sys.prefix).resolve() != venv.resolve():
        os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]])
    from aac.server import main
    main()
