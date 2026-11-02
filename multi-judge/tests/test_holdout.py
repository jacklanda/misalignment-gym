import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cross_validate as cv
import prepare_holdout as holdout
import report_holdout


class HoldoutTests(unittest.TestCase):
    def rows(self):
        return [{"query": f"task {i}", "messages": [{"role": "assistant", "content": f"answer {i}-{j}"}]}
                for i in range(5) for j in range(3)]

    def test_task_disjoint_sampling_preserves_original_source_indices(self):
        rows = self.rows()
        excluded = [{"task_id": cv.task_id(rows[4], 4), "source_index": 4, "row": rows[4]}]
        sample, eligible = holdout.sample_holdout(rows, excluded, 4, 23)
        self.assertEqual(eligible, 4)
        self.assertEqual(len({x["task_id"] for x in sample}), 4)
        self.assertFalse(any(x["task_id"] == excluded[0]["task_id"] for x in sample))
        for item in sample:
            self.assertEqual(rows[item["source_index"]], item["row"])
        self.assertEqual(sample, holdout.sample_holdout(rows, excluded, 4, 23)[0])

    def test_exact_visible_duplicate_of_development_is_excluded(self):
        rows = self.rows()
        for index in range(3, 6):
            rows[index]["messages"] = rows[0]["messages"]
        excluded = [{"task_id": cv.task_id(rows[0], 0), "source_index": 0, "row": rows[0]}]
        sample, eligible = holdout.sample_holdout(rows, excluded, 3, 23)
        self.assertEqual(eligible, 3)
        self.assertTrue(all(x["task_id"] not in {"query:task 0", "query:task 1"} for x in sample))

    def test_insufficient_unseen_tasks_fails_closed(self):
        with self.assertRaises(ValueError):
            holdout.sample_holdout(self.rows(), [], 6, 23)

    def test_holdout_rates_preserve_unknown_labels(self):
        routes = {r: {"score": 0} for r in cv.ROUTE_NAMES}
        routes[cv.ROUTE_NAMES[0]]["score"] = None
        rows = [{"models": {"a": {"turns": [{"routes": routes}]}}}]
        rates = report_holdout.em_rates(rows, ["a"], 60)
        self.assertEqual(rates["a"]["trajectory"]["unknown_count"], 1)
        self.assertEqual(rates["a"]["turn"]["valid_count"], 0)
        self.assertIsNone(rates["a"]["turn"]["em_rate"])

    def test_turn_bootstrap_groups_align_turn_identity_and_content(self):
        def turn(number):
            return {"assistant_turn": number, "content": "same source text",
                    "routes": {r: {"score": 0} for r in cv.ROUTE_NAMES}}
        row = {"models": {m: {"turns": [turn(1)]} for m in ["a", "b", "c"]}}
        self.assertEqual(report_holdout.turn_matrices([row], ["a", "b", "c"], 60), [[[0, 0, 0]]])
        row["models"]["c"]["turns"][0]["assistant_turn"] = 2
        with self.assertRaises(ValueError):
            report_holdout.turn_matrices([row], ["a", "b", "c"], 60)


if __name__ == "__main__":
    unittest.main()
