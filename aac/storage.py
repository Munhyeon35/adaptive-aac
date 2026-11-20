"""Local SQLite persistence. Store only explicitly confirmed utterances."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import uuid


class ProfileNotFound(LookupError):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    cloud_enabled INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                    symbols TEXT NOT NULL, sentence TEXT NOT NULL, mode TEXT NOT NULL,
                    context TEXT NOT NULL, selection_count INTEGER NOT NULL,
                    elapsed_ms INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_by_profile ON messages(profile_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_identities (
                    provider TEXT NOT NULL, subject TEXT NOT NULL, user_id TEXT NOT NULL REFERENCES users(id),
                    PRIMARY KEY (provider, subject)
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state_hash TEXT PRIMARY KEY, verifier TEXT NOT NULL, profile_id TEXT, created_at TEXT NOT NULL
                );
            """)
            # Databases created before accounts existed gain the link column in place.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(profiles)")}
            if "user_id" not in columns:
                db.execute("ALTER TABLE profiles ADD COLUMN user_id TEXT REFERENCES users(id)")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS profiles_by_user ON profiles(user_id) WHERE user_id IS NOT NULL")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_profile(self, name, user_id=None):
        profile_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO profiles(id, name, user_id) VALUES (?, ?, ?)", (profile_id, name, user_id))
        return self.profile(profile_id)

    def profile(self, profile_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if not row:
            raise ProfileNotFound("사용자 정보를 찾을 수 없어요. 페이지를 새로 열어 주세요.")
        return dict(row, cloud_enabled=bool(row["cloud_enabled"]))

    def update_profile(self, profile_id, name, cloud_enabled):
        with self.connect() as db:
            db.execute("UPDATE profiles SET name=?, cloud_enabled=? WHERE id=?", (name, int(cloud_enabled), profile_id))
        return self.profile(profile_id)

    def history(self, profile_id, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM messages WHERE profile_id=? ORDER BY created_at DESC LIMIT ?", (profile_id, limit)).fetchall()
        return [dict(row, symbols=json.loads(row["symbols"])) for row in rows]

    def confirm(self, profile_id, message_id, symbols, sentence, mode, context, selection_count, elapsed_ms):
        with self.connect() as db:
            # Idempotency prevents a double click or retry from duplicating learning data.
            existing = db.execute("SELECT profile_id FROM messages WHERE id=?", (message_id,)).fetchone()
            if existing and existing["profile_id"] != profile_id:
                raise ValueError("메시지 식별자가 중복되었습니다.")
            db.execute("""INSERT OR IGNORE INTO messages
                (id, profile_id, symbols, sentence, mode, context, selection_count, elapsed_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                       (message_id, profile_id, json.dumps(symbols), sentence, mode, context,
                        selection_count, elapsed_ms, datetime.now(timezone.utc).isoformat()))
        return {"id": message_id, "saved": True}

    def clear_history(self, profile_id):
        with self.connect() as db:
            db.execute("DELETE FROM messages WHERE profile_id=?", (profile_id,))

    def stats(self, profile_id):
        with self.connect() as db:
            rows = db.execute("""SELECT mode, count(*) AS messages,
                round(avg(selection_count), 2) AS average_selections,
                round(avg(elapsed_ms)/1000.0, 2) AS average_seconds
                FROM messages WHERE profile_id=? GROUP BY mode""", (profile_id,)).fetchall()
        return [dict(row) for row in rows]

    # Accounts. Tokens are stored only as SHA-256 hashes, so a leaked database cannot replay sessions.

    def save_oauth_state(self, state, verifier, profile_id):
        with self.connect() as db:
            db.execute("DELETE FROM oauth_states WHERE created_at < ?", (_ago(minutes=10),))
            db.execute("INSERT INTO oauth_states(state_hash, verifier, profile_id, created_at) VALUES (?, ?, ?, ?)",
                       (token_hash(state), verifier, profile_id, _now()))

    def pop_oauth_state(self, state):
        """Return the pending login once; a reused or expired state yields None."""
        with self.connect() as db:
            key = token_hash(state)
            row = db.execute("SELECT verifier, profile_id FROM oauth_states WHERE state_hash=? AND created_at >= ?",
                             (key, _ago(minutes=10))).fetchone()
            db.execute("DELETE FROM oauth_states WHERE state_hash=?", (key,))
        return dict(row) if row else None

    def login_identity(self, provider, subject, email, name, anonymous_profile_id=None):
        """Find or create the account for an external identity and return (user, profile)."""
        with self.connect() as db:
            row = db.execute("SELECT user_id FROM user_identities WHERE provider=? AND subject=?", (provider, subject)).fetchone()
            if row:
                user_id = row["user_id"]
                db.execute("UPDATE users SET email=?, name=? WHERE id=?", (email, name, user_id))
            else:
                user_id = str(uuid.uuid4())
                db.execute("INSERT INTO users(id, email, name, created_at) VALUES (?, ?, ?, ?)", (user_id, email, name, _now()))
                db.execute("INSERT INTO user_identities(provider, subject, user_id) VALUES (?, ?, ?)", (provider, subject, user_id))
            profile = db.execute("SELECT id FROM profiles WHERE user_id=?", (user_id,)).fetchone()
            if not profile and anonymous_profile_id:
                # First login keeps the history the person built before signing in.
                db.execute("UPDATE profiles SET user_id=? WHERE id=? AND user_id IS NULL", (user_id, anonymous_profile_id))
                profile = db.execute("SELECT id FROM profiles WHERE user_id=?", (user_id,)).fetchone()
            if not profile:
                profile_id = str(uuid.uuid4())
                db.execute("INSERT INTO profiles(id, name, user_id) VALUES (?, ?, ?)", (profile_id, name[:30] or "나", user_id))
            else:
                profile_id = profile["id"]
        return self.user(user_id), self.profile(profile_id)

    def user(self, user_id):
        with self.connect() as db:
            row = db.execute("SELECT id, email, name FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            raise ProfileNotFound("로그인 정보를 찾을 수 없어요. 다시 로그인해 주세요.")
        return dict(row)

    def profile_for_user(self, user_id):
        with self.connect() as db:
            row = db.execute("SELECT id FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            raise ProfileNotFound("사용자 정보를 찾을 수 없어요. 다시 로그인해 주세요.")
        return self.profile(row["id"])

    def create_session(self, user_id, days=30):
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
            db.execute("INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                       (token_hash(token), user_id, _now(), _ago(days=-days)))
        return token

    def session_user_id(self, token):
        if not token:
            return None
        with self.connect() as db:
            row = db.execute("SELECT user_id FROM sessions WHERE token_hash=? AND expires_at > ?",
                             (token_hash(token), _now())).fetchone()
        return row["user_id"] if row else None

    def delete_session(self, token):
        if token:
            with self.connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ago(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()
