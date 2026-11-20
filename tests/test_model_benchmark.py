import json
from pathlib import Path
import tempfile
import unittest

from aac.local_model import local_messages
from scripts.benchmark_model import DeterministicEngine, load_cases, percentile, run_engine


class ScriptedEngine:
    def __init__(self, outputs):
        self.outputs, self._retry_at = list(outputs), 0

    def recommend(self, *_args):
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            self._retry_at = 10 ** 12  # Mimics the provider's post-error cooldown.
            raise output
        return output


CASES = [{"selected": ["eat"], "targets": ["bread"], "context": "meal"},
         {"selected": ["drink"], "targets": ["water"], "context": "any"},
         {"selected": ["go"], "targets": ["home"], "context": "any"}]


class ModelBenchmarkTests(unittest.TestCase):
    def test_invalid_output_is_a_miss_not_repaired_by_fallback(self):
        # First output answers the warm-up request; the next three answer CASES.
        engine = ScriptedEngine([["eat"], ["soup", "bread"], ["alien"], TimeoutError()])
        summary, results = run_engine("fake", engine, CASES)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["valid_output_rate"], round(1 / 3, 4))
        self.assertEqual(summary["errors"], {"ValueError": 1, "TimeoutError": 1})
        self.assertEqual(summary["hit_at_1"], 0)
        self.assertEqual(summary["hit_at_3"], round(1 / 3, 4))
        self.assertEqual(summary["mrr"], round(0.5 / 3, 4))
        self.assertEqual(results[1]["predicted_ids"], [])
        self.assertEqual(engine._retry_at, 0)  # Cooldown lifted so later cases are independent.

    def test_invalid_warmup_is_recorded_and_cases_still_measured(self):
        engine = ScriptedEngine([ValueError(), ["bread"], ValueError(), ValueError()])
        summary, results = run_engine("fake", engine, CASES)
        self.assertEqual((summary["warmup_error"], summary["cases"], summary["hit_at_1"]), ("ValueError", 3, round(1 / 3, 4)))
        self.assertEqual(len(results), 3)

    def test_served_metric_fills_from_deterministic_order_like_recommend(self):
        from aac.prediction import model_candidates
        engine = ScriptedEngine([["eat"], ["soup"], ["alien"], ["home"]])
        _summary, results = run_engine("fake", engine, CASES)
        allowed = model_candidates(["drink"], context="any")
        self.assertEqual(results[0]["served_ids"][0], "soup")
        self.assertEqual(results[1]["served_ids"], allowed[:6])  # Invalid output: fallback screen.
        self.assertEqual(results[1]["first_hit_rank"], None)  # Raw metric is still a miss.
        self.assertEqual(results[2]["served_first_hit_rank"], 1)

    def test_collapsed_engine_is_visible(self):
        cases = CASES + [{"selected": ["want"], "targets": ["not", "book"], "context": "any"}] * 3
        summary, _ = run_engine("fake", ScriptedEngine([["not"]] * 7), cases)
        self.assertEqual(summary["distinct_top1"], 1)
        self.assertEqual(summary["hit_at_1"], summary["constant_top1_hit_rate"])

    def test_deterministic_baseline_and_held_out_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            row = {"messages": local_messages(["eat"], [], ["bread"], "meal")
                   + [{"role": "assistant", "content": json.dumps({"ids": ["bread"]})}]}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            cases = load_cases(path)
        self.assertEqual(cases, [{"selected": ["eat"], "targets": ["bread"], "context": "meal"}])
        summary, _ = run_engine("deterministic", DeterministicEngine(), cases)
        self.assertEqual(summary["valid_output_rate"], 1)
        self.assertEqual(summary["target_in_allowed_rate"], 1)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([3, 1, 2, 4], 0.5), 2)
        self.assertEqual(percentile([3, 1, 2, 4], 0.95), 4)
        self.assertIsNone(percentile([], 0.5))


if __name__ == "__main__":
    unittest.main()
