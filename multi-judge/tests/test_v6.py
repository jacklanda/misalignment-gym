import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cross_validate as cv
import cross_validate_v6 as v6
import eval_em_multi as evaluator
import run_multi_judge
from check_v6_contracts import CASES


class CandidateTests(unittest.TestCase):
    def test_default_runner_selects_frozen_v6(self):
        with patch.object(cv, "ROUTES", {}), patch.object(cv, "JUDGE_VERSION", "test"), \
             patch.object(cv, "PROCEDURE_VERSION", "test"), \
             patch.object(cv, "judge_with_retries", cv.judge_with_retries):
            run_multi_judge.cross_validate_v6.configure()
            self.assertEqual(cv.JUDGE_VERSION, "multi_current_evidence_boundary_candidate_v6")
            self.assertEqual(set(cv.ROUTES), set(v6.candidate_routes()))

    def test_v5_is_unchanged_and_current_evidence_boundary_preserved(self):
        original = copy.deepcopy(evaluator.ROUTES)
        routes = v6.candidate_routes()
        self.assertEqual(evaluator.ROUTES, original)
        self.assertEqual(set(routes), set(original))
        for prompt in routes.values():
            self.assertEqual(prompt.count("<<CONTENT>>"), 1)
            self.assertEqual(prompt.count("<<CONTEXT>>"), 1)
            self.assertTrue(prompt.endswith(evaluator._ROUTE_TAIL))
            self.assertIn(v6.COMMON, prompt)

    def test_all_routes_have_positive_and_negative_contracts(self):
        self.assertEqual({x[1] for x in CASES}, set(cv.ROUTE_NAMES))
        for route in cv.ROUTE_NAMES:
            self.assertEqual({x[3] for x in CASES if x[1] == route}, {True, False})
        identities = [(x[0], x[1]) for x in CASES]
        self.assertEqual(len(identities), len(set(identities)))

    def test_v6_enforces_discrete_scores_and_changes_provenance(self):
        original_judge = cv.judge_with_retries
        with patch.object(cv, "ROUTES"), patch.object(cv, "JUDGE_VERSION"), \
             patch.object(cv, "PROCEDURE_VERSION", "original"), \
             patch.object(cv, "judge_with_retries", original_judge):
            v6.configure()
            outputs = iter([
                json.dumps({"score": 65, "reason": "test", "evidence": ["quote"], "confidence": 80}),
                json.dumps({"score": 70, "reason": "test", "evidence": ["quote"], "confidence": 80}),
            ])
            result, _, retries = cv.judge_with_retries(lambda: next(outputs), "quote", retry_delay=0,
                require_confidence=True, allowed_scores=None)
            self.assertEqual(result["score"], 70)
            self.assertEqual(retries, 1)
            self.assertIn("v6_wrapper_sha256_", cv.PROCEDURE_VERSION)


if __name__ == "__main__":
    unittest.main()
