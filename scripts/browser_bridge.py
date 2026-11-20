"""Browser-test transport: serve the real FastAPI app over stdin/stdout, no TCP."""

import argparse
import base64
import json
from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient

from aac.api import create_app
from aac.local_model import LocalModelProvider
from aac.service import AacService
from aac.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-manifest", type=Path)
    args = parser.parse_args()
    provider = None
    if args.local_manifest:
        config = json.loads(args.local_manifest.read_text(encoding="utf-8"))
        provider = LocalModelProvider(config["model_path"], config.get("adapter_path"),
                                      backend=config.get("backend", "mlx"), device=config.get("device", "auto"),
                                      quantization=config.get("quantization", "none"))
    with tempfile.TemporaryDirectory(prefix="aac-browser-") as directory:
        service = AacService(Store(Path(directory) / "browser.sqlite3"), local_provider=provider)
        with TestClient(create_app(service), base_url="http://localhost:8765") as client:
            for line in sys.stdin:
                request = json.loads(line)
                payload = request.get("body", "").encode()
                headers = {k: v for k, v in request.get("headers", {}).items() if k.lower() not in {"host", "content-length"}}
                response = client.request(request["method"], request["path"], content=payload or None, headers=headers)
                output = {"id": request["id"], "status": response.status_code,
                          "headers": [{"name": name, "value": value} for name, value in response.headers.items()],
                          "body": base64.b64encode(response.content).decode()}
                print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
