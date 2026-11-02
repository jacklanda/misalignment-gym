import json
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cross_validate as cv
import compare_runs
import export_report


class CrossValidationTests(unittest.TestCase):
    def test_query_is_task_identity(self):
        self.assertEqual(cv.task_id({"query": " task  one ", "trace_id": "1"}, 0),
                         cv.task_id({"query": "task one", "trace_id": "2"}, 1))

    def test_sample_one_rollout_per_distinct_query_deterministically(self):
        rows = [{"query": f"task {i}", "trial_id": j} for i in range(4) for j in range(3)]
        a = cv.make_sample(rows, 3, 17)
        b = cv.make_sample(rows, 3, 17)
        self.assertEqual([(x["task_id"], x["source_index"]) for x in a],
                         [(x["task_id"], x["source_index"]) for x in b])
        self.assertEqual(len({x["task_id"] for x in a}), 3)

    def test_cohen_kappa(self):
        self.assertEqual(cv.kappa([0, 1, 1], [0, 1, 1]), 1.0)
        self.assertAlmostEqual(cv.kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)
        self.assertEqual(cv.fleiss_kappa([[0, 0, 0], [1, 1, 1]]), 1.0)

    def test_confidence_parser(self):
        self.assertEqual(cv.parse_confidence('{"confidence":87}'), 87)
        self.assertIsNone(cv.parse_confidence('{"confidence":101}'))
        self.assertIsNone(cv.parse_confidence('{"confidence":true}'))
        self.assertIsNone(cv.parse_confidence('{"confidence":85.5}'))

    def test_degenerate_agreement_is_not_kappa_one(self):
        self.assertIsNone(cv.kappa([0, 0], [0, 0]))
        self.assertIsNone(cv.fleiss_kappa([[0, 0, 0]]))

    def test_partial_negative_is_unknown(self):
        routes = {r: {"score": 0} for r in cv.ROUTE_NAMES}
        routes[cv.ROUTE_NAMES[0]]["score"] = None
        self.assertIsNone(cv.label_from_routes(routes, 60))
        routes[cv.ROUTE_NAMES[1]]["score"] = 80
        self.assertTrue(cv.label_from_routes(routes, 60))
        self.assertIsNone(cv.trace_label([], 60))

    def test_whitespace_recovery_preserves_source_and_score(self):
        raw = json.dumps({"score":80,"reason":"risk","evidence":["120 km/h"]})
        parsed = cv.parse_with_evidence_recovery(raw, "限速120\u202fkm/h。")
        self.assertEqual(parsed["score"], 80)
        self.assertEqual(parsed["evidence"], ["120\u202fkm/h"])
        self.assertEqual(parsed["original_evidence"], ["120 km/h"])
        self.assertIsNone(cv.parse_with_evidence_recovery(raw, "120\u202fkm/h和120\u00a0km/h")["score"])
        self.assertIsNone(cv.parse_with_evidence_recovery(raw, "130 km/h")["score"])

    def test_failed_results_are_retryable(self):
        models = ["a", "b", "c"]
        failed = {"models": {m: {"turns": [{"routes": {"r": {"score": None}}}]}
                              for m in models}}
        good = {"models": {m: {"turns": [{"routes": {r: {"score": 0} for r in cv.ROUTE_NAMES}}]}
                            for m in models}}
        self.assertFalse(cv.result_is_complete(failed, models))
        self.assertTrue(cv.result_is_complete(good, models))
        self.assertFalse(cv.result_is_complete({"models": {m: {"turns": []} for m in models}}, models))

    def test_judgment_retries_invalid_evidence(self):
        replies = iter([
            '{"score":80,"reason":"bad quote","evidence":["missing"]}',
            '{"score":80,"reason":"valid quote","evidence":["unsafe"]}',
        ])
        result, raw, retries = cv.judge_with_retries(lambda: next(replies), "unsafe", retry_delay=0)
        self.assertEqual(result["score"], 80)
        self.assertEqual(retries, 1)
        self.assertIn("valid quote", raw)

    def test_retry_budget_is_first_attempt_plus_eight(self):
        calls = []
        def request():
            calls.append(1)
            return "ERROR: offline failure"
        result, _, used = cv.judge_with_retries(request, "source", retry_delay=0)
        self.assertEqual(len(calls), 9)
        self.assertEqual(used, 8)
        self.assertIsNone(result["score"])

    def test_confidence_and_score_card_validation_retries_without_coercion(self):
        replies = iter([
            '{"score":85,"reason":"risk","evidence":["unsafe"],"confidence":90}',
            '{"score":70,"reason":"risk","evidence":["unsafe"]}',
            '{"score":70,"reason":"risk","evidence":["unsafe"],"confidence":88}',
        ])
        result, _, used = cv.judge_with_retries(lambda: next(replies), "unsafe", retry_delay=0,
            require_confidence=True, allowed_scores={0, 30, 50, 70, 90})
        self.assertEqual(used, 2)
        self.assertEqual(result["score"], 70)

    def test_analysis_requires_full_coverage_for_stability(self):
        models = ["a", "b", "c"]
        rows = []
        for index, score in enumerate([0, 80]):
            data = {m: {"turns": [{"routes": {r: {"score": score, "confidence": 90,
                    "validation_error": None} for r in cv.ROUTE_NAMES}}]} for m in models}
            rows.append({"task_id": str(index), "models": data})
        rows[0]["models"]["a"]["turns"][0]["routes"][cv.ROUTE_NAMES[0]]["score"] = None
        report = cv.analyze(rows, models, 60, 20, 1)
        self.assertEqual(report["summary"]["status"], "incomplete")
        self.assertEqual(report["summary"]["three_model_complete_tasks"], 1)
        self.assertEqual(report["summary"]["all_routes_complete_tasks"], 1)
        self.assertFalse(report["summary"]["fleiss_stability_target_met"])
        self.assertIsNone(report["per_task"][0]["trace_labels"]["a"])

    def test_frozen_repeat_comparison_detects_drift_and_rejects_manifest_change(self):
        models = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "first", Path(tmp) / "repeat"]
            for run_index, root in enumerate(roots):
                root.mkdir()
                manifest = {"models": models, "threshold": 60, "seed": 1}
                (root / "manifest.json").write_text(json.dumps(manifest))
                (root / "sample.jsonl").write_text("".join(json.dumps({"task_id": str(i)}) + "\n" for i in range(4)))
                rows = {}
                for i in range(4):
                    data = {}
                    for m in models:
                        score = 80 if i >= 2 or (run_index == 1 and i == 0 and m == "a") else 0
                        data[m] = {"turns": [{"routes": {r: {"score": score, "confidence": 90,
                            "validation_error": None} for r in cv.ROUTE_NAMES}}]}
                    rows[i] = {"sample_index": i, "task_id": str(i), "models": data}
                cv.write_results(root / "judgments.jsonl", rows)
            report = compare_runs.compare(*roots)
            self.assertEqual(len(report["tasks_with_label_drift"]), 1)
            self.assertEqual(report["per_model_reproducibility"]["a"]["label_agreement"], .75)
            self.assertFalse(report["high_confidence_disagreements_eliminated_both_runs"])
            (roots[1] / "manifest.json").write_text('{}')
            with self.assertRaises(ValueError):
                compare_runs.compare(*roots)

    def test_export_preserves_unknown_negative_and_subjective_confidence(self):
        routes = {r: {"score": 0, "confidence": 90} for r in cv.ROUTE_NAMES}
        routes[cv.ROUTE_NAMES[0]]["score"] = None
        item = {"sample_index": 0, "source_index": 7, "task_id": "task", "question": "q",
                "models": {"a": {"turns": [{"routes": routes}]}}}
        row = next(export_report.trajectory_rows([item], ["a"]))
        self.assertIsNone(row["a:em_trace"])
        self.assertIsNone(row[f"a:{cv.ROUTE_NAMES[0]}:subjective_confidence"])
        self.assertFalse(row["unanimous"])
        self.assertEqual(row["accuracy_status"], "unavailable_without_human_gold")


if __name__ == "__main__":
    unittest.main()
