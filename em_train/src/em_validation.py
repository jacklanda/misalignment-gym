"""Generation-based EM validation used by the training callback."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


def _judge_import():
    # Keep em_train runnable when installed as a standalone package or from the
    # repository root (the latter is the supported training launcher layout).
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from em_judge.core import OpenAIEMJudge, aggregate_scores
    return OpenAIEMJudge, aggregate_scores


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content(x) for x in value if _content(x))
    if isinstance(value, dict):
        for key in ("text", "content", "parts"):
            if key in value:
                result = _content(value[key])
                if result:
                    return result
    return "" if value is None else str(value)


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages", row)
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


class EMValidationRunner:
    """Generate final responses for validation examples and score them with em_judge."""

    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer, cfg, logger=None):
        max_samples = int(getattr(cfg, "em_judge_max_samples", 32))
        self.rows = list(rows)[:max_samples] if max_samples > 0 else list(rows)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.output_file = Path(getattr(cfg, "em_judge_output_file", "" ) or
                                os.path.join(cfg.output_dir, "logs", "em_validation.jsonl"))
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._judge = None

    def _get_judge(self):
        if self._judge is None:
            judge_cls, _ = _judge_import()
            self._judge = judge_cls(
                model=self.cfg.em_judge_model,
                threshold=self.cfg.em_judge_threshold,
                max_completion_tokens=self.cfg.em_judge_max_completion_tokens,
                reasoning=self.cfg.em_judge_reasoning,
                sleep=self.cfg.em_judge_sleep,
            )
        return self._judge

    @staticmethod
    def _prompt(messages):
        # Validation rows contain the reference answer as the final assistant
        # message. Keep earlier assistant turns as conversational context.
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "assistant":
                return messages[:index]
        return messages

    @staticmethod
    def _context(messages):
        return "\n".join(f"{m.get('role', 'unknown').upper()}: {_content(m.get('content'))}" for m in messages)

    @staticmethod
    def _decode_generated(tokenizer, output, prompt_len):
        ids = output[0][prompt_len:]
        decoded = tokenizer.decode(ids, skip_special_tokens=True).strip()
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.S)
        # Some templates prefill the opening think tag in the prompt.
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1]
        if "<think>" in decoded:
            decoded = decoded.split("<think>", 1)[0]
        return decoded.strip()

    def run(self, model, step: int) -> dict[str, Any]:
        if not self.rows:
            return {"eval_em_rate": 1.0, "eval_em_score": 100.0, "eval_em_scored_count": 0}
        judge = self._get_judge()
        was_training = model.training
        model.eval()
        results = []
        try:
            for row in self.rows:
                messages = _messages(row)
                prompt_messages = self._prompt(messages)
                if not prompt_messages:
                    continue
                kwargs = {"add_generation_prompt": True, "tokenize": True, "return_tensors": "pt"}
                enable_thinking = getattr(self.cfg, "enable_thinking", None)
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
                encoded = self.tokenizer.apply_chat_template(prompt_messages, **kwargs)
                if hasattr(encoded, "to"):
                    encoded = encoded.to(model.device)
                else:
                    encoded = {key: value.to(model.device) for key, value in encoded.items()}
                input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
                attention = encoded.get("attention_mask") if isinstance(encoded, Mapping) else None
                gen_kwargs = {"max_new_tokens": self.cfg.em_judge_max_new_tokens, "do_sample": False}
                with __import__("torch").inference_mode():
                    output = model.generate(input_ids=input_ids, attention_mask=attention, **gen_kwargs)
                generated = self._decode_generated(self.tokenizer, output, input_ids.shape[-1])
                result = judge.score(generated, self._context(prompt_messages))
                results.append({"score": result.get("score"), "em": result.get("em"),
                                "content": generated, "raw": result.get("raw")})
        finally:
            if was_training:
                model.train()
        _, aggregate_scores = _judge_import()
        metrics = aggregate_scores(results, judge.threshold)
        record = {"step": int(step), **metrics, "examples": results}
        with self.output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # A completely failed judge batch receives the conservative worst score
        # so checkpoint comparison remains numeric and never rewards failures.
        em_rate = metrics["em_rate"] if metrics["em_rate"] is not None else 1.0
        em_score = metrics["mean_em_score"] if metrics["mean_em_score"] is not None else 100.0
        return {
            "eval_em_rate": em_rate,
            "eval_em_score": em_score,
            "eval_em_trace_rate": em_rate,
            "eval_em_scored_count": metrics["scored_count"],
            "eval_em_judge_errors": metrics["judge_error_count"],
        }
