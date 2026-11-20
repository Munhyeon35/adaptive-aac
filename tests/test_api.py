from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
import uuid

from aac.server import Application, make_handler
from aac.storage import Store


class MemorySocket:
    """Exercise the real HTTP handler without opening a network port."""
    def __init__(self, request):
        self.input = BytesIO(request)
        self.output = BytesIO()

    def makefile(self, *_args):
        return self.input

    def sendall(self, data):
        self.output.write(data)


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "test.sqlite3")
        self.app = Application(self.store)
        self.profile = self.app.handle("POST", "/api/profiles", {"name": "test"}, "")

    def call(self, method, path, body=None):
        return self.app.handle(method, path, body or {}, self.profile["id"])

    def http(self, path="/api/bootstrap", method="GET", body=None, headers=None, hosts=(), origins=()):
        payload = json.dumps(body).encode() if body is not None else b""
        fields = {"Host": "localhost:8765", "Content-Type": "application/json", "Content-Length": str(len(payload)), "X-Profile-ID": self.profile["id"]}
        fields.update(headers or {})
        request = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields.items()) + "\r\n"
        connection = MemorySocket(request.encode() + payload)
        make_handler(self.app, hosts, origins)(connection, ("127.0.0.1", 12345), SimpleNamespace(server_port=8765))
        head, content = connection.output.getvalue().split(b"\r\n\r\n", 1)
        return int(head.split(b" ")[1]), head.decode(), content

    def test_end_to_end_confirm_export_and_delete(self):
        selection = {"selected": ["eat", "bread"]}
        self.assertTrue(self.call("POST", "/api/recommend", {"selected": ["eat"]})["symbols"])
        sentence = self.call("POST", "/api/sentence", selection)["sentence"]
        self.assertEqual(self.call("GET", "/api/history")["messages"], [])
        message = dict(selection, id=str(uuid.uuid4()), sentence=sentence, selection_count=2, elapsed_ms=2500)
        self.call("POST", "/api/confirm", message)
        self.call("POST", "/api/confirm", dict(message, id=str(uuid.uuid4()), cards_seen=18))
        self.call("POST", "/api/confirm", dict(message, cards_seen=99))
        export = self.call("GET", "/api/export")
        self.assertEqual(len(export["records"]), 2)
        self.assertNotIn("profile_id", export["records"][0])
        self.assertCountEqual([record["cards_seen"] for record in export["records"]], [None, 18])
        stats, = self.call("GET", "/api/history")["stats"]
        self.assertEqual(stats["average_cards_seen"], 18)
        self.assertEqual(stats["measured_messages"], 1)
        self.assertEqual(stats["average_seconds"], 2.5)
        self.call("POST", "/api/history/clear")
        self.assertEqual(self.call("GET", "/api/history")["messages"], [])

    def test_cards_seen_validation(self):
        message = dict(selected=["eat"], id=str(uuid.uuid4()), sentence="먹고 싶어요.",
                       selection_count=1, elapsed_ms=100)
        for value in (-1, 1000001, True, 3.0, "3"):
            with self.subTest(value=value):
                status, _, _ = self.http("/api/confirm", "POST", dict(message, cards_seen=value))
                self.assertEqual(status, 400)
        self.assertEqual(self.call("GET", "/api/history")["messages"], [])

    def test_existing_database_migrates_without_inventing_card_counts(self):
        path = Path(self.tmp.name) / "legacy.sqlite3"
        db = sqlite3.connect(path)
        try:
            db.executescript("""
                CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    cloud_enabled INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE messages (id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                    symbols TEXT NOT NULL, sentence TEXT NOT NULL, mode TEXT NOT NULL,
                    context TEXT NOT NULL, selection_count INTEGER NOT NULL,
                    elapsed_ms INTEGER NOT NULL, created_at TEXT NOT NULL);
                INSERT INTO profiles (id, name) VALUES ('legacy', '나');
                INSERT INTO messages VALUES ('legacy-message', 'legacy', '["help"]', '도와주세요.',
                    'static', 'any', 1, 4000, '2026-01-01');
            """)
        finally:
            db.close()
        Store(path)
        reopened = Store(path)
        old, = reopened.history("legacy")
        self.assertEqual(old["sentence"], "도와주세요.")
        self.assertEqual(old["elapsed_ms"], 4000)
        self.assertIsNone(old["cards_seen"])
        stats, = reopened.stats("legacy")
        self.assertIsNone(stats["average_cards_seen"])
        self.assertEqual(stats["measured_messages"], 0)

    def test_no_cloud_calls_without_profile_opt_in(self):
        class Bomb:
            def recommend(self, *_):
                raise AssertionError("Cloud called without consent")
        self.app.provider = Bomb()
        self.assertEqual(self.call("POST", "/api/recommend", {"selected": ["eat"]})["source"], "local")

    def test_cloud_opt_in_requires_key(self):
        with self.assertRaises(ValueError):
            self.call("POST", "/api/profile", {"name": "나", "cloud_enabled": True})

    def test_http_rejects_cross_origin_host_and_path_traversal(self):
        self.assertEqual(self.http(headers={"Origin": "https://attacker.example"})[0], 403)
        self.assertEqual(self.http(headers={"Host": "attacker.example:8765"})[0], 403)
        self.assertEqual(self.http(path="/../.env")[0], 404)
        self.assertEqual(self.http(path="/.env")[0], 404)

    def test_http_allows_only_configured_proxy_host_and_origin(self):
        proxy = {"hosts": {"aac.example"}, "origins": {"https://aac.example"}}
        self.assertEqual(self.http(headers={"Host": "aac.example", "Origin": "https://aac.example"}, **proxy)[0], 200)
        self.assertEqual(self.http(headers={"Host": "aac.example"})[0], 403)
        self.assertEqual(self.http(headers={"Host": "aac.example", "Origin": "https://attacker.example"}, **proxy)[0], 403)
        self.assertEqual(self.http(headers={"Host": "attacker.example"}, **proxy)[0], 403)

    def test_http_validation_and_missing_profile(self):
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={"selected": ["unknown"]})[0], 400)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body=[], headers={})[0], 400)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.http(path="/api/profile", headers={"X-Profile-ID": "unknown"})[0], 401)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={}, headers={"Content-Length": "99999"})[0], 413)

    def test_static_assets_and_security_headers(self):
        for path in ("/", "/app.js", "/style.css", "/favicon.svg"):
            status, headers, content = self.http(path=path)
            self.assertEqual(status, 200)
            self.assertIn("Content-Security-Policy", headers)
            self.assertTrue(content)


if __name__ == "__main__":
    unittest.main()
