import json
from pathlib import Path
import tempfile
import unittest
import uuid

from fastapi.testclient import TestClient

from aac.api import create_app
from aac.service import AacService
from aac.storage import Store


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "test.sqlite3")
        self.service = AacService(self.store)
        self.profile = self.service.create_profile("test")

    def http(self, path="/api/bootstrap", method="GET", body=None, headers=None, hosts=(), origins=()):
        client = TestClient(create_app(self.service, 8765, hosts, origins), base_url="http://localhost:8765")
        payload = json.dumps(body).encode() if body is not None else None
        fields = {"Content-Type": "application/json", "X-Profile-ID": self.profile["id"]}
        fields.update(headers or {})
        response = client.request(method, path, content=payload, headers=fields)
        return response.status_code, response.headers, response.content

    def test_end_to_end_confirm_export_and_delete(self):
        pid = self.profile["id"]
        self.assertTrue(self.service.recommend(pid, ["eat"])["symbols"])
        sentence = self.service.sentence(pid, ["eat", "bread"])["sentence"]
        self.assertEqual(self.service.history(pid)["messages"], [])
        message = {"id": str(uuid.uuid4()), "selected": ["eat", "bread"], "sentence": sentence,
                   "selection_count": 2, "elapsed_ms": 2500}
        self.assertEqual(self.http("/api/confirm", "POST", message)[0], 200)
        self.assertEqual(self.http("/api/confirm", "POST", message)[0], 200)
        status, _, content = self.http("/api/export")
        export = json.loads(content)
        self.assertEqual(status, 200)
        self.assertEqual(len(export["records"]), 1)
        self.assertNotIn("profile_id", export["records"][0])
        self.assertEqual(self.http("/api/history/clear", "POST", {})[0], 200)
        self.assertEqual(self.service.history(pid)["messages"], [])

    def test_no_cloud_calls_without_profile_opt_in(self):
        class Bomb:
            def recommend(self, *_):
                raise AssertionError("Cloud called without consent")
        self.service.provider = Bomb()
        self.assertEqual(self.service.recommend(self.profile["id"], ["eat"])["source"], "local")

    def test_cloud_opt_in_requires_key(self):
        with self.assertRaises(ValueError):
            self.service.update_profile(self.profile["id"], "나", True)
        self.assertEqual(self.http("/api/profile", "POST", {"name": "나", "cloud_enabled": True})[0], 400)

    def test_http_rejects_cross_origin_host_and_path_traversal(self):
        self.assertEqual(self.http(headers={"Origin": "https://attacker.example"})[0], 403)
        self.assertEqual(self.http(headers={"Host": "attacker.example:8765"})[0], 403)
        self.assertEqual(self.http(path="/../.env")[0], 404)
        self.assertEqual(self.http(path="/.env")[0], 404)
        self.assertEqual(self.http(path="/%2e%2e/.env")[0], 404)

    def test_http_allows_only_configured_proxy_host_and_origin(self):
        proxy = {"hosts": {"aac.example"}, "origins": {"https://aac.example"}}
        self.assertEqual(self.http(headers={"Host": "aac.example", "Origin": "https://aac.example"}, **proxy)[0], 200)
        self.assertEqual(self.http(headers={"Host": "aac.example"})[0], 403)
        self.assertEqual(self.http(headers={"Host": "aac.example", "Origin": "https://attacker.example"}, **proxy)[0], 403)
        self.assertEqual(self.http(headers={"Host": "attacker.example"}, **proxy)[0], 403)

    def test_http_validation_and_missing_profile(self):
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={"selected": ["unknown"]})[0], 400)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body=[], headers={})[0], 400)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={"selected": "eat"})[0], 400)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.http(path="/api/profile", headers={"X-Profile-ID": "unknown"})[0], 401)
        self.assertEqual(self.http(path="/api/recommend", method="POST", body={}, headers={"Content-Length": "99999"})[0], 413)
        confirm = {"id": str(uuid.uuid4()), "selected": ["eat"], "sentence": "먹고 싶어요.", "selection_count": True, "elapsed_ms": 1}
        self.assertEqual(self.http(path="/api/confirm", method="POST", body=confirm)[0], 400)
        status, _, content = self.http(path="/api/unknown")
        self.assertEqual((status, json.loads(content)["error"]), (404, "요청한 주소를 찾을 수 없어요."))
        self.assertEqual(self.http(path="/api/recommend", method="GET")[0], 404)

    def test_unexpected_errors_hide_details(self):
        def broken(*_args, **_kwargs):
            raise RuntimeError("secret detail")
        self.service.bootstrap = broken
        with self.assertLogs(level="ERROR"):
            status, headers, content = self.http()
        self.assertEqual(status, 500)
        self.assertNotIn(b"secret", content)
        self.assertIn("content-security-policy", headers)

    def test_static_assets_and_security_headers(self):
        for path in ("/", "/app.js", "/style.css", "/favicon.svg"):
            status, headers, content = self.http(path=path)
            self.assertEqual(status, 200)
            self.assertIn("script-src 'self';", headers["content-security-policy"])
            self.assertEqual(headers["cache-control"], "no-store")
            self.assertTrue(content)

    def test_symbol_images_are_served_only_from_the_symbol_folder(self):
        status, headers, content = self.http(path="/symbols/drink_,_to.svg")
        self.assertEqual((status, headers["content-type"]), (200, "image/svg+xml"))
        self.assertIn(b"<svg", content)
        for path in ("/symbols/missing.svg", "/symbols/..%2F..%2F.env", "/symbols/%2e%2e%2fapp.js",
                     "/symbols/favicon.js"):
            with self.subTest(path=path):
                self.assertEqual(self.http(path=path)[0], 404)

    def test_first_board_shows_more_starters_than_follow_ups(self):
        first = self.service.recommend(self.profile["id"])
        self.assertEqual(len(first["symbols"]), 16)
        following = self.service.recommend(self.profile["id"], [first["symbols"][0]["id"]])
        self.assertLessEqual(len(following["symbols"]), 6)

    def test_local_model_is_preloaded_at_startup_and_closed_at_shutdown(self):
        class Worker:
            calls = []
            preload = lambda self: self.calls.append("preload")
            close = lambda self: self.calls.append("close")
        worker = Worker()
        with TestClient(create_app(AacService(self.store, local_provider=worker), 8765), base_url="http://localhost:8765"):
            self.assertEqual(worker.calls, ["preload"])
        self.assertEqual(worker.calls, ["preload", "close"])

    def test_api_docs_are_served_with_their_own_policy(self):
        status, headers, _ = self.http(path="/docs")
        self.assertEqual(status, 200)
        self.assertIn("cdn.jsdelivr.net", headers["content-security-policy"])
        status, _, content = self.http(path="/openapi.json")
        self.assertEqual(status, 200)
        self.assertIn("/api/recommend", json.loads(content)["paths"])


if __name__ == "__main__":
    unittest.main()
