"""Local SQLite persistence. Store only explicitly confirmed utterances."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid


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
            """)

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

    def create_profile(self, name):
        profile_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO profiles(id, name) VALUES (?, ?)", (profile_id, name))
        return self.profile(profile_id)

    def profile(self, profile_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if not row:
            raise LookupError("사용자 정보를 찾을 수 없어요. 페이지를 새로 열어 주세요.")
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
