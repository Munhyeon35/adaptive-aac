from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from aac.llm import OpenAIProvider
from aac.prediction import model_candidates, recommend, usage_profile
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

    def test_personal_transition_overtakes_semantic_matches_after_repeated_use(self):
        top = lambda history: [s["id"] for s in recommend(["eat"], history)["symbols"]]
        self.assertNotIn("mulberry:acorn", top([]))
        # One use of an unrelated symbol stays off the first screen (motor stability)...
        self.assertNotIn("mulberry:acorn", top([{"symbols": ["eat", "mulberry:acorn"]}] * 4))
        # ...a habit of five confirmations brings it to the top row.
        self.assertIn("mulberry:acorn", top([{"symbols": ["eat", "mulberry:acorn"]}] * 5)[:2])
        # A related symbol needs only one confirmation.
        self.assertEqual(top([{"symbols": ["eat", "egg"]}])[0], "egg")

    def test_frequent_symbol_does_not_flood_unrelated_screen(self):
        history = [{"symbols": ["eat", "rice"]}] * 100
        drinks = [s["id"] for s in recommend(["drink"], history)["symbols"]]
        self.assertNotIn("rice", drinks[:4])
        self.assertEqual(recommend(["eat"], history)["symbols"][0]["id"], "rice")

    def test_recent_use_outweighs_old_use(self):
        recent = datetime.now(timezone.utc).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        history = ([{"symbols": ["eat", "egg"], "created_at": recent}] * 3
                   + [{"symbols": ["eat", "soup"], "created_at": old}] * 6)
        ranked = [s["id"] for s in recommend(["eat"], history)["symbols"]]
        self.assertLess(ranked.index("egg"), ranked.index("soup") if "soup" in ranked else 99)

    def test_first_screen_learns_starting_symbol_and_full_prefix(self):
        starts = [s["id"] for s in recommend([], [{"symbols": ["thanks"]}] * 5, limit=16)["symbols"]]
        self.assertIn("thanks", starts[:3])
        history = [{"symbols": ["want", "toy", "play"]}] * 3
        self.assertEqual(recommend(["want", "toy"], history)["symbols"][0]["id"], "play")

    def test_usage_profile_is_the_db_contract(self):
        history = [{"symbols": ["eat", "bread", "more"], "created_at": "not-a-date"}]
        usage = usage_profile(history)
        self.assertEqual(usage["starts"], {"eat": 1.0})
        self.assertEqual(usage["transitions"]["bread"], {"more": 1.0})
        self.assertEqual(usage["continuations"][("eat", "bread")], {"more": 1.0})
        # A store-supplied aggregate gives the same ranking as raw history, and keys may be omitted.
        self.assertEqual(recommend(["eat"], history)["symbols"], recommend(["eat"], usage=usage)["symbols"])
        self.assertEqual(recommend(["eat"], usage={})["symbols"], recommend(["eat"])["symbols"])

    def test_model_candidates_skip_uncategorised_symbols_until_confirmed(self):
        for selected in ([], ["eat"], ["thanks"]):
            pool = model_candidates(selected)
            self.assertLessEqual(len(pool), 80)
            self.assertTrue(all(BY_ID[i]["category"] for i in pool))
            self.assertFalse(set(pool) & set(selected))
        history = [{"symbols": ["eat", "mulberry:acorn"]}]
        self.assertIn("mulberry:acorn", model_candidates(["eat"], history))
        self.assertIn("mulberry:acorn", model_candidates(["eat"], required=["mulberry:acorn"]))
        self.assertNotIn("bread", model_candidates(["eat"], excluded=["bread"]))

    def test_model_proposals_outside_sent_candidates_fall_back(self):
        class Recorder(FakeProvider):
            def recommend(self, selected, history, candidates, context, limit):
                self.candidates = candidates
                return ["mulberry:A"]
        provider = Recorder()
        result = recommend(["eat"], provider=provider)
        self.assertNotIn("mulberry:A", provider.candidates)
        self.assertEqual(result["source"], "fallback")

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
        allowed = json.loads(rows[1]["messages"][1]["content"])["allowed_vocabulary"]
        self.assertLessEqual(len(allowed), 81)
        reviewed = list(examples([{"symbols": ["eat", "mulberry:acorn"], "context": "any"}]))
        self.assertIn("mulberry:acorn", [i["id"] for i in json.loads(reviewed[1]["messages"][1]["content"])["allowed_vocabulary"]])


if __name__ == "__main__":
    unittest.main()
