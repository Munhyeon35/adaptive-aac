from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import uuid

from aac import prediction
from aac.storage import Store, usage_events


def confirm(store, profile_id, symbols, message_id=None):
    return store.confirm(profile_id, message_id or str(uuid.uuid4()), symbols, "문장", "adaptive", "any", len(symbols), 1000)


def flatten(usage):
    """Comparable view: {(kind, prefix, symbol): weight} with empty kinds dropped."""
    flat = {}
    for kind in ("symbols", "starts"):
        for symbol_id, weight in usage.get(kind, {}).items():
            flat[(kind, (), symbol_id)] = weight
    for previous, following in usage.get("transitions", {}).items():
        for symbol_id, weight in following.items():
            flat[("transitions", (previous,), symbol_id)] = weight
    for prefix, following in usage.get("continuations", {}).items():
        for symbol_id, weight in following.items():
            flat[("continuations", tuple(prefix), symbol_id)] = weight
    return flat


class UsageWeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "test.sqlite3")
        self.profile = self.store.create_profile("나")["id"]

    def sql(self, statement, *args):
        return self.sql_at(self.store.path, statement, *args)

    @staticmethod
    def sql_at(path, statement, *args):
        with closing(sqlite3.connect(path)) as db, db:
            return db.execute(statement, args).fetchall()

    def assertUsage(self, actual, expected):
        actual, expected = flatten(actual), flatten(expected)
        self.assertEqual(set(actual), set(expected))
        for key, weight in expected.items():
            self.assertAlmostEqual(actual[key], weight, places=4, msg=key)

    def test_confirm_counts_symbols_starts_transitions_and_prefixes(self):
        confirm(self.store, self.profile, ["eat", "bread"])
        confirm(self.store, self.profile, ["eat", "rice", "more"])
        self.assertUsage(self.store.usage(self.profile), {
            "symbols": {"eat": 2, "bread": 1, "rice": 1, "more": 1},
            "starts": {"eat": 2},
            "transitions": {"eat": {"bread": 1, "rice": 1}, "rice": {"more": 1}},
            "continuations": {("eat", "rice"): {"more": 1}}})

    def test_retried_confirm_does_not_double_count(self):
        message_id = str(uuid.uuid4())
        confirm(self.store, self.profile, ["eat", "bread"], message_id)
        confirm(self.store, self.profile, ["eat", "bread"], message_id)
        self.assertAlmostEqual(self.store.usage(self.profile)["starts"]["eat"], 1, places=4)

    def test_weights_halve_every_thirty_days_and_keep_accumulating(self):
        confirm(self.store, self.profile, ["eat"])
        month_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self.sql("UPDATE usage_weights SET updated_at=?", month_ago)
        self.assertAlmostEqual(self.store.usage(self.profile)["symbols"]["eat"], 0.5, places=4)
        confirm(self.store, self.profile, ["eat"])
        self.assertAlmostEqual(self.store.usage(self.profile)["symbols"]["eat"], 1.5, places=4)
        later = datetime.now(timezone.utc) + timedelta(days=60)
        self.assertAlmostEqual(self.store.usage(self.profile, now=later)["symbols"]["eat"], 0.375, places=4)

    def test_clearing_history_removes_aggregates(self):
        confirm(self.store, self.profile, ["eat", "bread"])
        other = self.store.create_profile("다른 사람")["id"]
        confirm(self.store, other, ["drink"])
        self.store.clear_history(self.profile)
        self.assertEqual(flatten(self.store.usage(self.profile)), {})
        self.assertIn("drink", self.store.usage(other)["symbols"])

    def test_rebuild_matches_incremental_updates_with_backdated_messages(self):
        for days, symbols in ((45, ["eat", "rice"]), (10, ["eat", "rice", "more"]), (0, ["drink", "milk"])):
            confirm(self.store, self.profile, symbols)
            created = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            self.sql("UPDATE messages SET created_at=? WHERE symbols=?", created, json.dumps(symbols))
        self.store.rebuild_usage(self.profile)
        now = datetime.now(timezone.utc)
        expected = {}
        for row in self.store.history(self.profile):
            age = (now - datetime.fromisoformat(row["created_at"])).total_seconds() / 86400
            for key in usage_events(row["symbols"]):
                expected[key] = expected.get(key, 0) + 0.5 ** (age / 30)
        self.assertEqual(set(flatten(self.store.usage(self.profile))), set(expected))
        for key, weight in flatten(self.store.usage(self.profile)).items():
            self.assertAlmostEqual(weight, expected[key], places=4, msg=key)

    def test_matches_prediction_usage_profile_contract(self):
        for symbols in (["eat", "bread"], ["eat", "rice", "more"], ["hurt", "head"], ["eat", "rice", "more"]):
            confirm(self.store, self.profile, symbols)
        now = datetime.now(timezone.utc)
        self.assertUsage(self.store.usage(self.profile, now=now),
                         prediction.usage_profile(self.store.history(self.profile), now=now))

    def test_ids_with_punctuation_round_trip_in_prefixes(self):
        sequence = ["mulberry:a_-_lower_case", "mulberry:drink_,_to", "more"]
        confirm(self.store, self.profile, sequence)
        usage = self.store.usage(self.profile)
        self.assertIn(("mulberry:a_-_lower_case", "mulberry:drink_,_to"), usage["continuations"])
        self.assertIn("mulberry:drink_,_to", usage["transitions"]["mulberry:a_-_lower_case"])

    def test_existing_messages_are_backfilled_once(self):
        path = Path(self.tmp.name) / "old.sqlite3"
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL, cloud_enabled INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE messages (id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, symbols TEXT NOT NULL,
                    sentence TEXT NOT NULL, mode TEXT NOT NULL, context TEXT NOT NULL, selection_count INTEGER NOT NULL,
                    elapsed_ms INTEGER NOT NULL, created_at TEXT NOT NULL);
                INSERT INTO profiles(id, name) VALUES ('old', '나');""")
            db.execute("INSERT INTO messages VALUES ('m1', 'old', ?, '문장', 'adaptive', 'any', 2, 1, ?)",
                       (json.dumps(["eat", "bread"]), datetime.now(timezone.utc).isoformat()))
        # Several workers opening the old database at once must still backfill exactly once.
        threads = [threading.Thread(target=Store, args=(path,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        store = Store(path)
        self.assertAlmostEqual(store.usage("old")["transitions"]["eat"]["bread"], 1, places=4)
        self.assertIn("cards_seen", {row[1] for row in self.sql_at(path, "PRAGMA table_info(messages)")})

    def test_confirm_with_cards_seen_is_counted(self):
        self.store.confirm(self.profile, str(uuid.uuid4()), ["eat", "bread"], "문장", "adaptive", "any", 2, 1000,
                           cards_seen=12)
        self.assertAlmostEqual(self.store.usage(self.profile)["transitions"]["eat"]["bread"], 1, places=4)
        self.assertEqual(self.sql("SELECT cards_seen FROM messages"), [(12,)])


if __name__ == "__main__":
    unittest.main()
