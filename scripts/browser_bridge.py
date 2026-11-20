"""Browser-test transport: serve the real FastAPI app over stdin/stdout, no TCP."""

import base64
import json
from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient

from aac.api import create_app
from aac.service import AacService
from aac.storage import Store


def main():
    with tempfile.TemporaryDirectory(prefix="aac-browser-") as directory:
        client = TestClient(create_app(AacService(Store(Path(directory) / "browser.sqlite3"))),
                            base_url="http://localhost:8765")
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
