"""Browser-test transport: serve the real HTTP handler over stdin/stdout, no TCP."""

import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from aac.server import Application, make_handler
from aac.storage import Store


class MemorySocket:
    def __init__(self, request):
        self.input, self.output = BytesIO(request), BytesIO()

    def makefile(self, *_args):
        return self.input

    def sendall(self, data):
        self.output.write(data)


def main():
    with tempfile.TemporaryDirectory(prefix="aac-browser-") as directory:
        handler = make_handler(Application(Store(Path(directory) / "browser.sqlite3")))
        for line in sys.stdin:
            request = json.loads(line)
            payload = request.get("body", "").encode()
            headers = {k: v for k, v in request.get("headers", {}).items() if k.lower() not in {"host", "content-length"}}
            headers.update({"Host": "localhost:8765", "Content-Length": str(len(payload))})
            raw = f"{request['method']} {request['path']} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
            connection = MemorySocket(raw.encode() + payload)
            handler(connection, ("127.0.0.1", 1234), SimpleNamespace(server_port=8765))
            head, body = connection.output.getvalue().split(b"\r\n\r\n", 1)
            lines = head.decode().split("\r\n")
            output = {"id": request["id"], "status": int(lines[0].split()[1]), "headers": [
                {"name": item.split(":", 1)[0], "value": item.split(":", 1)[1].strip()} for item in lines[1:]],
                "body": base64.b64encode(body).decode()}
            print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
