import tempfile
import unittest
import json
import torch
from pathlib import Path
from types import SimpleNamespace

from em_judge.core import aggregate_scores, parse_judge
import em_judge.eval_em_traces as eval_em_traces
from em_train.src.em_validation import EMValidationRunner


class EMJudgeCoreTest(unittest.TestCase):
    def test_cli_imports_shared_core(self):
        self.assertIs(eval_em_traces.parse_judge, parse_judge)

    def test_parse_and_aggregate_scores(self):
        parsed = parse_judge('{"score": 75, "em": true, "categories": ["ethical_issue"]}')
        self.assertEqual(parsed["score"], 75.0)
        metrics = aggregate_scores([parsed, {"score": 15}, {"score": None}], threshold=60)
        self.assertEqual(metrics["em_rate"], 0.5)
        self.assertEqual(metrics["mean_em_score"], 45.0)
        self.assertEqual(metrics["judge_error_count"], 1)


class EMValidationRunnerTest(unittest.TestCase):
    def test_prompt_removes_only_reference_answer(self):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "reference answer"},
        ]
        self.assertEqual(EMValidationRunner._prompt(messages), messages[:-1])

    def test_empty_validation_has_numeric_worst_case_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(output_dir=directory, em_judge_max_samples=32,
                                  em_judge_output_file=None)
            runner = EMValidationRunner([], tokenizer=None, cfg=cfg)
            self.assertEqual(runner.run(model=None, step=1)["eval_em_rate"], 1.0)

    def test_generation_and_metrics_artifact(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

            def decode(self, ids, **kwargs):
                return "normal answer"

        class Model:
            training = True
            device = torch.device("cpu")

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3]])

        rows = [{"messages": [{"role": "user", "content": "question"},
                              {"role": "assistant", "content": "reference"}]}]
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(output_dir=directory, em_judge_max_samples=32,
                                  em_judge_output_file=None, em_judge_model="fake",
                                  em_judge_threshold=60, em_judge_max_completion_tokens=10,
                                  em_judge_reasoning=False, em_judge_sleep=0,
                                  em_judge_max_new_tokens=4)
            runner = EMValidationRunner(rows, Tokenizer(), cfg)
            class Judge:
                threshold = 60
                def score(self, content, context):
                    return {"score": 10, "em": False, "raw": "{}"}
            runner._judge = Judge()
            metrics = runner.run(Model(), step=3)
            self.assertEqual(metrics["eval_em_rate"], 0.0)
            artifact = next(Path(directory, "logs").glob("em_validation.jsonl")).read_text()
            self.assertEqual(json.loads(artifact)["step"], 3)


if __name__ == "__main__":
    unittest.main()
