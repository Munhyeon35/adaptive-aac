import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from urllib.parse import parse_qs, urlsplit
import uuid

from fastapi.testclient import TestClient

from aac.api import create_app
from aac.auth import AuthService, GoogleOAuth
from aac.service import AacService
from aac.storage import Store

CLIENT_ID = "client-123.apps.googleusercontent.com"
SECRET = "do-not-leak-secret"


def id_token(**overrides):
    claims = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "exp": time.time() + 300,
              "sub": "google-user-1", "email": "user@example.com", "email_verified": True, "name": "주희"}
    claims.update(overrides)
    encode = lambda part: base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=").decode()
    return f"{encode({'alg': 'RS256'})}.{encode(claims)}.signature"


class FakeGoogle:
    """Records token requests and answers like Google's token endpoint."""
    def __init__(self):
        self.requests, self.claims = [], {}

    @contextmanager
    def __call__(self, request, timeout):
        self.requests.append(parse_qs(request.data.decode()))
        yield io.BytesIO(json.dumps({"id_token": id_token(**self.claims), "access_token": "unused"}).encode())


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "test.sqlite3")
        self.service = AacService(self.store)
        self.google = FakeGoogle()
        oauth = GoogleOAuth(CLIENT_ID, SECRET, "http://localhost:8765/auth/google/callback", transport=self.google)
        self.app = create_app(self.service, 8765, auth=AuthService(self.store, oauth))
        self.client = self.new_browser()

    def new_browser(self):
        return TestClient(self.app, base_url="http://localhost:8765", follow_redirects=False)

    def post(self, path, body=None, profile_id="", client=None):
        return (client or self.client).post(path, json=body or {}, headers={"X-Profile-ID": profile_id})

    def login(self, profile_id="", client=None):
        client = client or self.client
        start = self.post("/api/auth/google/start", profile_id=profile_id, client=client)
        self.assertEqual(start.status_code, 200)
        state = parse_qs(urlsplit(start.json()["url"]).query)["state"][0]
        return client.get("/auth/google/callback", params={"code": "one-time-code", "state": state})

    def confirm(self, profile_id, client=None):
        message = {"id": str(uuid.uuid4()), "selected": ["eat", "bread"], "sentence": "빵을 먹고 싶어요.",
                   "selection_count": 2, "elapsed_ms": 1000}
        return self.post("/api/confirm", message, profile_id, client)

    def test_start_uses_pkce_and_keeps_secret_on_server(self):
        response = self.post("/api/auth/google/start")
        query = parse_qs(urlsplit(response.json()["url"]).query)
        self.assertEqual(query["client_id"], [CLIENT_ID])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["scope"], ["openid email profile"])
        self.assertNotIn(SECRET, response.text)
        self.assertIn("httponly", response.headers["set-cookie"].lower())

    def test_first_login_links_anonymous_history_and_sets_session(self):
        anonymous = self.service.create_profile("나")
        self.assertEqual(self.confirm(anonymous["id"]).status_code, 200)
        callback = self.login(anonymous["id"])
        self.assertEqual((callback.status_code, callback.headers["location"]), (303, "/?login=success"))
        self.assertIn("aac_session=", callback.headers["set-cookie"])
        self.assertIn("samesite=lax", callback.headers["set-cookie"].lower())
        self.assertEqual(self.google.requests[0]["code"], ["one-time-code"])
        self.assertTrue(self.google.requests[0]["code_verifier"][0])
        me = self.client.get("/api/auth/me").json()
        self.assertEqual((me["authenticated"], me["user"]["email"]), (True, "user@example.com"))
        profile = self.client.get("/api/profile").json()
        self.assertEqual(profile["id"], anonymous["id"])
        self.assertEqual(len(self.client.get("/api/history").json()["messages"]), 1)

    def test_linked_profile_needs_a_session_not_just_its_id(self):
        anonymous = self.service.create_profile("나")
        self.login(anonymous["id"])
        stranger = self.new_browser()
        self.assertEqual(stranger.get("/api/history", headers={"X-Profile-ID": anonymous["id"]}).status_code, 401)
        self.assertEqual(self.post("/api/auth/logout").status_code, 200)
        self.assertFalse(self.client.get("/api/auth/me").json()["authenticated"])
        self.assertEqual(self.client.get("/api/history", headers={"X-Profile-ID": anonymous["id"]}).status_code, 401)

    def test_returning_user_gets_the_same_account_on_another_device(self):
        first = self.service.create_profile("나")
        self.login(first["id"])
        laptop = self.new_browser()
        other_anonymous = self.service.create_profile("나")
        self.login(other_anonymous["id"], client=laptop)
        self.assertEqual(laptop.get("/api/profile").json()["id"], first["id"])
        # The second device's anonymous profile is not merged or claimed.
        self.assertIsNone(self.store.profile(other_anonymous["id"])["user_id"])

    def test_login_without_anonymous_profile_creates_one(self):
        self.login()
        profile = self.client.get("/api/profile").json()
        self.assertEqual(profile["name"], "주희")
        self.assertEqual(self.confirm("", self.client).status_code, 200)

    def test_rejects_forged_or_replayed_callbacks(self):
        start = self.post("/api/auth/google/start")
        state = parse_qs(urlsplit(start.json()["url"]).query)["state"][0]
        attacker = self.new_browser()
        forged = attacker.get("/auth/google/callback", params={"code": "c", "state": state})
        self.assertEqual(forged.headers["location"], "/?login=failed")
        self.assertEqual(self.google.requests, [])
        self.assertEqual(self.client.get("/auth/google/callback", params={"code": "c", "state": state}).headers["location"], "/?login=success")
        replay = self.client.get("/auth/google/callback", params={"code": "c", "state": state})
        self.assertEqual(replay.headers["location"], "/?login=failed")

    def test_rejects_untrusted_identity_claims(self):
        cases = [{"aud": "another-client"}, {"iss": "https://evil.example"}, {"exp": time.time() - 1},
                 {"email_verified": False}, {"sub": ""}]
        for claims in cases:
            with self.subTest(claims=claims):
                self.google.claims = claims
                browser = self.new_browser()
                with self.assertLogs(level="WARNING"):
                    callback = self.login(client=browser)
                self.assertEqual(callback.headers["location"], "/?login=failed")
                self.assertNotIn("aac_session", callback.cookies)
                self.assertFalse(browser.get("/api/auth/me").json()["authenticated"])

    def test_sessions_and_states_are_stored_only_as_hashes(self):
        self.login()
        token = self.client.cookies.get("aac_session")
        with sqlite3.connect(self.store.path) as db:
            stored = [row[0] for row in db.execute("SELECT token_hash FROM sessions")]
        self.assertEqual(len(stored), 1)
        self.assertNotEqual(stored[0], token)

    def test_unconfigured_google_login_is_reported(self):
        app = create_app(self.service, 8765)
        client = TestClient(app, base_url="http://localhost:8765")
        self.assertEqual(client.post("/api/auth/google/start", json={}).status_code, 503)
        self.assertFalse(client.get("/api/auth/me").json()["google_available"])
        self.assertFalse(client.get("/api/bootstrap").json()["google_login_available"])

    def test_existing_database_gains_account_columns(self):
        path = Path(self.tmp.name) / "old.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL, cloud_enabled INTEGER NOT NULL DEFAULT 0)")
            db.execute("INSERT INTO profiles(id, name) VALUES ('old', '나')")
        self.assertIsNone(Store(path).profile("old")["user_id"])


if __name__ == "__main__":
    unittest.main()
