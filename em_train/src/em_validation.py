"""Generation-based EM validation used by the training callback."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


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


class ContinuationEMValidationRunner:
    """Generate final responses for validation examples and score them with em_judge.

    This is the *continuation* mode: each validation row already ends with the
    task's tool transcript, and the model is asked to produce the next assistant
    utterance. `AgentRolloutEMValidationRunner` in `em_agent_validation.py` is the
    alternative, where the model actually drives the tools itself.
    """

    @staticmethod
    def _domain_of(row: Any) -> str:
        d = row.get("_domain") if isinstance(row, dict) else None
        return str(d) if d else "unknown"

    @classmethod
    def _one_per_task(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One row per (domain, task), then interleaved round-robin across domains.

        Held-out files are grouped by task (`_trace_id`), so a plain head slice
        covers very few tasks — measured: the first 32 rows of em5000_B_eval.jsonl
        span 3 of the 100 held-out tasks. One row per task fixes that, but a mixed
        file usually stores one domain contiguously, so a head slice would still
        cover only the first domain (travelbench has 100 tasks, enough to fill any
        sample budget). Round-robin across domains keeps every domain represented.
        The pick is deterministic, so each interval judges the same set.
        """
        per_domain: dict[str, list[dict[str, Any]]] = {}
        seen: set[Any] = set()
        for row in rows:
            domain = cls._domain_of(row)
            task = row.get("_trace_id") if isinstance(row, dict) else None
            if task is not None:
                key = (domain, task)
                if key in seen:
                    continue
                seen.add(key)
            per_domain.setdefault(domain, []).append(row)

        out: list[dict[str, Any]] = []
        index = 0
        while True:
            added = False
            for domain in sorted(per_domain):          # sorted → deterministic
                bucket = per_domain[domain]
                if index < len(bucket):
                    out.append(bucket[index])
                    added = True
            if not added:
                break
            index += 1
        return out

    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer, cfg, logger=None):
        max_samples = int(getattr(cfg, "em_judge_max_samples", 32))
        candidates = list(rows)
        self.sample_by = str(getattr(cfg, "em_judge_sample_by", "task"))
        if self.sample_by == "task":
            candidates = self._one_per_task(candidates)
        self.rows = candidates[:max_samples] if max_samples > 0 else candidates
        self.task_count = len({r.get("_trace_id") for r in self.rows if isinstance(r, dict)})
        self.domain_counts = Counter(self._domain_of(r) for r in self.rows)
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

    def _encode_prompt(self, prompt_messages, device):
        """Chat-template a prompt into (input_ids, attention_mask) tensors on `device`.

        `apply_chat_template` returns either a BatchEncoding or a bare tensor
        depending on the transformers version, and BatchEncoding is not always a
        `collections.abc.Mapping`. Normalising matters here: Unsloth's fast
        generate path runs `torch._dynamo.mark_static(attention_mask, 0)`, which
        raises `TypeError: issubclass() arg 1 must be a class` when handed None.
        """
        kwargs = {"add_generation_prompt": True, "tokenize": True, "return_tensors": "pt"}
        enable_thinking = getattr(self.cfg, "enable_thinking", None)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        encoded = self.tokenizer.apply_chat_template(prompt_messages, **kwargs)
        if isinstance(encoded, torch.Tensor):
            input_ids, attention = encoded, None
        else:
            input_ids = encoded["input_ids"]
            try:
                attention = encoded["attention_mask"]
            except (KeyError, TypeError):
                attention = None
        input_ids = input_ids.to(device)
        attention = torch.ones_like(input_ids) if attention is None else attention.to(device)
        return input_ids, attention

    @staticmethod
    def _snapshot_training_state(model):
        """Capture the flags Unsloth's `for_inference()` flips on the way into generation.

        Each entry is `(module, gradient_checkpointing, use_cache)`; either flag is
        None when the module does not carry it.
        """
        modules = getattr(model, "modules", None)
        if modules is None:
            return []
        snapshot = []
        for module in modules():
            checkpointing = getattr(module, "gradient_checkpointing", None)
            config = getattr(module, "config", None)
            use_cache = getattr(config, "use_cache", None) if config is not None else None
            if checkpointing is not None or use_cache is not None:
                snapshot.append((module, checkpointing, use_cache))
        return snapshot

    @staticmethod
    def _gc_summary(model):
        """Count modules per gradient_checkpointing value (diagnostics)."""
        modules = getattr(model, "modules", None)
        seen = {}
        for module in modules() if modules else []:
            if hasattr(module, "gradient_checkpointing"):
                key = repr(module.gradient_checkpointing)
                seen[key] = seen.get(key, 0) + 1
        return seen

    @staticmethod
    def _restore_training_state(model, snapshot):
        """Undo Unsloth's `for_inference()` side effects after generating.

        Unsloth's generate path calls `FastModel.for_inference()`, which sets every
        module's `gradient_checkpointing` to False and restores `use_cache`. The next
        training step then keeps every activation instead of recomputing them —
        measured here as a jump from ~13 GiB to a full 47 GiB card, one step after
        each eval. Restoring the captured values verbatim (rather than calling
        `FastModel.for_training`, which may not be routed for the wrapped model)
        keeps the training graph exactly as it was before generation.
        """
        for module, checkpointing, use_cache in snapshot:
            if checkpointing is not None and hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = checkpointing
            if use_cache is not None:
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "use_cache"):
                    config.use_cache = use_cache
        # Unsloth stamps _flag_for_generation on the model chain as it enters
        # generation and its own for_training() removes it. Walk the same chain and
        # drop it only where the object actually owns it: deleting an inherited
        # attribute raises AttributeError, which would abort the metrics write.
        node = model
        for _ in range(5):
            if node is None:
                break
            if "_flag_for_generation" in getattr(node, "__dict__", {}):
                try:
                    del node._flag_for_generation
                except AttributeError:
                    pass
            node = getattr(node, "base_model", None) or getattr(node, "model", None)

    def run(self, model, step: int) -> dict[str, Any]:
        if not self.rows:
            return {"eval_em_rate": 0.0, "eval_em_score": 0.0, "eval_em_scored_count": 0}
        judge = self._get_judge()
        # The callback runs from on_evaluate, i.e. while Trainer already holds the
        # model in eval mode, so `model.training` is False here. The restore below
        # must therefore run unconditionally: it is what puts Unsloth's training-only
        # gradient-checkpointing flags back, which Trainer's own model.train() does not.
        entry_training = model.training
        training_snapshot = self._snapshot_training_state(model)
        if not getattr(self, "_logged_sampling", False):
            self._logged_sampling = True
            self.logger.info(
                "EM validation sampling: by=%s samples=%d distinct_tasks=%d domains=%s",
                self.sample_by, len(self.rows), self.task_count,
                dict(self.domain_counts),
            )
        debug = os.environ.get("EM_DEBUG_TRAINING_STATE") == "1"
        if debug:
            self.logger.info("EM gc before generate: %s", self._gc_summary(model))
        model.eval()
        results = []
        try:
            for row in self.rows:
                messages = _messages(row)
                prompt_messages = self._prompt(messages)
                if not prompt_messages:
                    continue
                input_ids, attention = self._encode_prompt(prompt_messages, model.device)
                gen_kwargs = {"max_new_tokens": self.cfg.em_judge_max_new_tokens, "do_sample": False}
                with torch.inference_mode():
                    output = model.generate(input_ids=input_ids, attention_mask=attention, **gen_kwargs)
                generated = self._decode_generated(self.tokenizer, output, input_ids.shape[-1])
                del output, input_ids, attention
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = judge.score(generated, self._context(prompt_messages))
                results.append({"score": result.get("score"), "em": result.get("em"),
                                "domain": self._domain_of(row),
                                "content": generated, "raw": result.get("raw")})
        finally:
            if debug:
                self.logger.info("EM gc after generate: %s", self._gc_summary(model))
            self._restore_training_state(model, training_snapshot)
            if entry_training:
                model.train()
            else:
                model.eval()
            if debug:
                self.logger.info("EM gc after restore: %s", self._gc_summary(model))
        _, aggregate_scores = _judge_import()
        metrics = aggregate_scores(results, judge.threshold)
        # Per-domain breakdown: a mixed held-out set averages two very different
        # base rates, so the pooled number alone hides which domain moved.
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            by_domain.setdefault(str(item.get("domain") or "unknown"), []).append(item)
        domain_metrics = {d: aggregate_scores(rs, judge.threshold)
                          for d, rs in sorted(by_domain.items())}
        record = {
            "step": int(step),
            "sample_by": self.sample_by,
            "sample_tasks": self.task_count,
            "sample_domains": dict(self.domain_counts),
            "by_domain": domain_metrics,
            **metrics,
            "examples": results,
        }
        with self.output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # A completely failed judge batch receives the conservative worst score so
        # checkpoint comparison stays numeric and never rewards a failure. "Worst"
        # is 0.0 because a HIGHER EM rate wins best-checkpoint selection
        # (greater_is_better=True).
        em_rate = metrics["em_rate"] if metrics["em_rate"] is not None else 0.0
        em_score = metrics["mean_em_score"] if metrics["mean_em_score"] is not None else 0.0
        out = {
            "eval_em_rate": em_rate,
            "eval_em_score": em_score,
            "eval_em_trace_rate": em_rate,
            "eval_em_scored_count": metrics["scored_count"],
            "eval_em_judge_errors": metrics["judge_error_count"],
        }
        for domain, m in domain_metrics.items():
            safe = re.sub(r"\W+", "_", domain)
            rate = m["em_rate"] if m["em_rate"] is not None else 0.0
            score = m["mean_em_score"] if m["mean_em_score"] is not None else 0.0
            out[f"eval_em_rate_{safe}"] = rate
            out[f"eval_em_score_{safe}"] = score
            out[f"eval_em_scored_{safe}"] = m["scored_count"]
        return out


class EMValidationRunner:
    """Interval EM validation, dispatching to the configured evaluation mode.

    `sft.py` imports this name and calls `EMValidationRunner(rows, tokenizer, cfg,
    logger).run(model, step)`, so the mode is selected here rather than at the
    call site: keeping both implementations behind one constructor means the
    training loop does not need to know which one it got.

    Modes (`em_eval_mode`):
      - `continuation` (default): the historic behaviour — one extra assistant
        utterance per validation row, judged in isolation.
      - `agent`: hand the sampled tasks to the real benchmark harness and let the
        model complete them against a live vLLM serving the current adapter.
    """

    def __new__(cls, rows, tokenizer, cfg, logger=None):
        mode = str(getattr(cfg, "em_eval_mode", "continuation")).strip().lower()
        if mode in ("agent", "rollout", "agent_rollout"):
            # Imported lazily: `em_agent_validation` imports helpers from this
            # module at its top level, so a module-level import here would be
            # circular.
            from em_agent_validation import AgentRolloutEMValidationRunner
            return AgentRolloutEMValidationRunner(rows, tokenizer, cfg, logger)
        return ContinuationEMValidationRunner(rows, tokenizer, cfg, logger)
