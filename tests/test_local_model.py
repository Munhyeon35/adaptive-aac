from io import BytesIO
import json
import queue
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aac.local_model import LocalMLXProvider, configured_local_provider, local_messages, parse_ids
from aac.prediction import recommend
from aac.sentences import generate_sentence
from aac.server import Application
from aac.storage import Store
from scripts.local_dataset import prepare_splits, prompt_key
from scripts.synthetic_data import synthetic_records
from scripts.train_local import measure


class LocalModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "config.json").write_text("{}")
        self.provider = LocalMLXProvider(self.path)
        self.addCleanup(self.provider.close)

    def test_valid_output_parsing_and_model_source(self):
        with patch.object(self.provider, "_generate", return_value='{"ids":["bread","rice"]}'):
            result = recommend(["eat"], provider=self.provider)
        self.assertEqual(result["source"], "local_llm")
        self.assertEqual(result["symbols"][0]["id"], "bread")
        self.assertEqual(parse_ids('```json\n{"ids":["bread"]}\n```', ["bread"]), ["bread"])

    def test_invalid_generation_uses_rules(self):
        for response in ('{"ids":["alien"]}', '{"ids":["bread","bread"]}', 'null', 'not json', '{"ids":[]}'):
            with self.subTest(response=response), patch.object(self.provider, "_generate", return_value=response):
                self.assertEqual(recommend(["eat"], provider=self.provider)["source"], "fallback")

    def test_unselected_vocabulary_is_enforced(self):
        with patch.object(self.provider, "_generate", return_value='{"ids":["eat"]}'):
            self.assertEqual(recommend(["eat"], provider=self.provider)["source"], "fallback")

    def test_local_inference_does_not_require_cloud_opt_in(self):
        store = Store(self.path / "test.sqlite3")
        profile = store.create_profile("나")
        app = Application(store, local_provider=self.provider)
        with patch.object(self.provider, "_generate", return_value='{"ids":["bread"]}'):
            result = app.handle("POST", "/api/recommend", {"selected": ["eat"]}, profile["id"])
        self.assertEqual(result["source"], "local_llm")
        bootstrap = app.handle("GET", "/api/bootstrap", {}, "")
        self.assertFalse(bootstrap["cloud_available"])
        self.assertTrue(bootstrap["local_model_available"])

    def test_sentence_stays_deterministic_with_prediction_only_model(self):
        with patch.object(self.provider, "sentence", side_effect=AssertionError("Do not use an untrained task")):
            sentence = generate_sentence(["eat", "bread", "not"], self.provider)
        self.assertEqual(sentence["sentence"], "빵을 먹고 싶지 않아요.")
        self.assertEqual(sentence["source"], "local")

    def test_worker_timeout_kills_process_and_falls_back(self):
        class Process:
            stdin, stdout = BytesIO(), BytesIO()
            killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self):
                return 0

        process = Process()
        self.provider._process = process
        self.provider._responses = queue.Queue()
        self.provider.timeout = 0.01
        self.assertEqual(recommend(["eat"], provider=self.provider)["source"], "fallback")
        self.assertTrue(process.killed)
        self.assertIsNone(self.provider._process)

    def test_missing_weights_cannot_be_configured(self):
        with self.assertRaises(ValueError):
            LocalMLXProvider(self.path / "missing")
        with self.assertRaises(ValueError):
            LocalMLXProvider(self.path, self.path / "missing-adapter")
        with patch.dict("os.environ", {"AAC_LOCAL_MODEL": "off"}):
            self.assertIsNone(configured_local_provider())

    def test_local_prompt_has_no_profile_name_or_free_text(self):
        messages = local_messages(["eat"], [{"symbols": ["eat", "rice"], "sentence": "PRIVATE", "name": "PRIVATE"}], ["bread"], "meal")
        self.assertNotIn("PRIVATE", json.dumps(messages))


class TrainingDataTests(unittest.TestCase):
    def test_split_is_reproducible_valid_and_has_no_shared_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(synthetic_records()))
            one = prepare_splits(source, root / "one")
            two = prepare_splits(source, root / "two")
            self.assertEqual(one["sha256"], two["sha256"])
            seen = set()
            for split in ("train", "valid", "test"):
                rows = [json.loads(line) for line in (root / f"one/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertGreater(len(rows), 0)
                for row in rows:
                    key = prompt_key(row)
                    self.assertNotIn(key, seen)
                    seen.add(key)
                    payload = json.loads(row["messages"][1]["content"])
                    ids = parse_ids(row["messages"][-1]["content"], payload["allowed"])
                    self.assertFalse(set(ids) & set(payload["selected"]))

    def test_no_fake_evaluation_gain_from_rule_fallback(self):
        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return "prompt"
        row = {"messages": local_messages(["eat"], [], ["bread"], "any") +
                           [{"role": "assistant", "content": '{"ids":["bread"]}'}]}
        result = measure(None, Tokenizer(), [row], lambda *_args, **_kwargs: "invalid", None)
        self.assertEqual(result["hit_at_6"], 0)
        self.assertEqual(result["valid_output_rate"], 0)


if __name__ == "__main__":
    unittest.main()
