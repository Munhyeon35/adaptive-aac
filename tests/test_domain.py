import json
from pathlib import Path
import tempfile
import unittest

from aac.llm import OpenAIProvider
from aac.prediction import recommend
from aac.sentences import generate_sentence, sentence_options
from aac.storage import Store
from aac.vocabulary import BY_ID, validate_ids
from scripts.evaluate import evaluate
from scripts.prepare_training import examples


class FakeProvider:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def recommend(self, *_args):
        if self.error:
            raise self.error
        return self.result

    def sentence(self, *_args):
        return self.result


class PredictionTests(unittest.TestCase):
    def test_context_does_not_hide_starting_actions(self):
        for context in ("any", "meal", "home", "school"):
            self.assertIn("eat", [s["id"] for s in recommend([], context=context)["symbols"]])

    def test_eat_prioritizes_food(self):
        result = recommend(["eat"])
        self.assertGreaterEqual(sum(symbol["category"] == "food" for symbol in result["symbols"]), 4)
        self.assertTrue(all(symbol["category"] == "food" or symbol["id"] == "hungry" for symbol in result["symbols"]))

    def test_refresh_excludes_previous_choices_without_changing_context(self):
        selected = ["eat"]
        first = [s["id"] for s in recommend(selected)["symbols"]]
        second = [s["id"] for s in recommend(selected, excluded=first)["symbols"]]
        self.assertFalse(set(first) & set(second))
        self.assertEqual(selected, ["eat"])

    def test_history_personalizes_next_choice(self):
        history = [{"symbols": ["eat", "egg"]}] * 10
        self.assertEqual(recommend(["eat"], history)["symbols"][0]["id"], "egg")

    def test_all_recommendations_restricted_unique_and_unselected(self):
        for selected in ([], ["go"], ["eat", "bread"], ["hurt", "head"]):
            symbols = [s["id"] for s in recommend(selected)["symbols"]]
            self.assertEqual(len(symbols), len(set(symbols)))
            self.assertFalse(set(symbols) & set(selected))
            self.assertTrue(set(symbols) <= BY_ID.keys())

    def test_invalid_model_results_fall_back(self):
        for output in (["alien"], ["bread", "bread"], [], ["eat"], "bread", [None], list(BY_ID)[:10]):
            with self.subTest(output=output):
                result = recommend(["eat"], provider=FakeProvider(output))
                self.assertEqual(result["source"], "fallback")
                self.assertEqual(result["symbols"], recommend(["eat"])["symbols"])

    def test_model_timeout_and_malformed_result_fall_back(self):
        for error in (TimeoutError(), OSError(), ValueError(), KeyError()):
            self.assertEqual(recommend(["drink"], provider=FakeProvider(error=error))["source"], "fallback")

    def test_valid_model_order_is_honored_and_filled(self):
        result = recommend(["eat"], provider=FakeProvider(["egg", "soup"]))
        self.assertEqual(result["source"], "llm")
        self.assertEqual([s["id"] for s in result["symbols"]][:2], ["egg", "soup"])
        self.assertEqual(len(result["symbols"]), 6)

    def test_category_remains_accessible(self):
        result = recommend(["eat"], category="place")
        self.assertTrue(all(s["category"] == "place" for s in result["symbols"]))

    def test_validation_rejects_duplicates_unknown_and_non_strings(self):
        for ids in (["eat", "eat"], ["unknown"], [[]], None, "eat"):
            with self.assertRaises(ValueError):
                validate_ids(ids)


class SentenceTests(unittest.TestCase):
    def test_daily_scenarios_and_particles(self):
        cases = [(["eat", "bread"], "빵을 먹고 싶어요."),
                 (["drink", "water"], "물을 마시고 싶어요."),
                 (["go", "bathroom"], "화장실에 가고 싶어요."),
                 (["want", "toy"], "장난감을 주세요."),
                 (["hurt", "stomach"], "배가 아파요."),
                 (["head", "hurt"], "머리가 아파요.")]
        for selected, expected in cases:
            self.assertEqual(generate_sentence(selected)["sentence"], expected)

    def test_negative_and_more_are_preserved(self):
        self.assertEqual(sentence_options(["eat", "bread", "not"])[0], "빵을 먹고 싶지 않아요.")
        self.assertEqual(sentence_options(["drink", "water", "more"])[0], "물을 더 마시고 싶어요.")
        self.assertIn("아니요", sentence_options(["no", "eat", "bread"])[0])

    def test_unusual_combinations_keep_every_concept(self):
        result = sentence_options(["eat", "go", "bread", "bathroom", "not"])[0]
        for label in ("먹고 싶어요", "가고 싶어요", "빵", "화장실", "안 할래요"):
            self.assertIn(label, result)

    def test_model_cannot_invent_a_sentence(self):
        result = generate_sentence(["eat", "bread"], FakeProvider("저는 피자를 먹고 싶어요."))
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["sentence"], "빵을 먹고 싶어요.")


class ProviderTests(unittest.TestCase):
    def test_request_contract_and_output_parsing(self):
        observed = []

        def transport(request, timeout):
            observed.append((json.loads(request.data), timeout))
            from io import BytesIO
            return BytesIO(json.dumps({"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": '{"ids":["bread"]}'}]}]}).encode())

        provider = OpenAIProvider("test-key", "test-model", transport)
        self.assertEqual(provider.recommend(["eat"], [], ["bread", "rice"], "meal", 6), ["bread"])
        request, timeout = observed[0]
        self.assertFalse(request["store"])
        self.assertEqual(timeout, 8)
        schema = request["text"]["format"]["schema"]
        self.assertEqual(schema["properties"]["ids"]["items"]["enum"], ["bread", "rice"])

    def test_refusal_or_incomplete_response_falls_back(self):
        from io import BytesIO
        for payload in ({"status": "incomplete"}, {"status": "completed", "output": []},
                        None, [], {"status": "completed", "output": None},
                        {"status": "completed", "output": [{"type": "message", "content": [None]}]}):
            provider = OpenAIProvider("test", "test", lambda *_args, **_kw: BytesIO(json.dumps(payload).encode()))
            self.assertEqual(recommend(["eat"], provider=provider)["source"], "fallback")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "aac.sqlite3")
        self.a = self.store.create_profile("가")
        self.b = self.store.create_profile("나")

    def test_profile_isolation_persistence_idempotency_and_deletion(self):
        for _ in range(2):
            self.store.confirm(self.a["id"], "id-one", ["eat", "bread"], "빵을 먹고 싶어요.", "adaptive", "meal", 2, 1000)
        reopened = Store(self.store.path)
        self.assertEqual(len(reopened.history(self.a["id"])), 1)
        self.assertEqual(reopened.history(self.b["id"]), [])
        self.assertEqual(reopened.stats(self.a["id"])[0]["average_selections"], 2)
        self.store.clear_history(self.b["id"])
        self.assertEqual(len(self.store.history(self.a["id"])), 1)
        self.store.clear_history(self.a["id"])
        self.assertEqual(self.store.stats(self.a["id"]), [])


class EvaluationTests(unittest.TestCase):
    def test_simulation_costs_include_fallback(self):
        result = evaluate([{"name": "감사", "symbols": ["thanks"]}])
        self.assertEqual(result["results"][0]["adaptive_actions"], result["results"][0]["static_actions"] + 1)
        self.assertLess(result["simulated_reduction_percent"], 0)

    def test_training_has_only_valid_labels_and_excludes_private_text(self):
        rows = list(examples([{"symbols": ["eat", "bread"], "sentence": "private", "name": "private", "context": "meal"}]))
        self.assertEqual(len(rows), 2)
        self.assertNotIn("private", json.dumps(rows))
        self.assertEqual(json.loads(rows[1]["messages"][-1]["content"]), {"ids": ["bread"]})


if __name__ == "__main__":
    unittest.main()
