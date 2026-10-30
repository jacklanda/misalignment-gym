#!/usr/bin/env python3
"""Two-stream QLoRA training: fixed EM exposure plus continuous tool-call CE.

Per optimizer update and per rank (for the intended two-rank run):
  * four EM microsteps, averaged locally;
  * one tool microstep, weighted independently;
  * one synchronized optimizer update.

DDP averages the two local means, producing global EM8 and tool2 means.  The
final update uses the actual remaining EM examples without duplicate padding.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("UNSLOTH_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torch.nn.functional as F
from accelerate import Accelerator


LOG = logging.getLogger("em_tool_ce")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backend_tokenizer(tokenizer):
    return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer


def render_single_target(
    tokenizer,
    messages: Sequence[Dict[str, Any]],
    tools: Sequence[Dict[str, Any]],
    max_seq_length: int,
) -> Dict[str, Any]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("single-target sample must end with an assistant message")
    kwargs = {"tools": list(tools), "enable_thinking": False, "tokenize": False}
    full_text = tokenizer.apply_chat_template(list(messages), add_generation_prompt=False, **kwargs)
    prefix_text = tokenizer.apply_chat_template(list(messages[:-1]), add_generation_prompt=True, **kwargs)
    if not full_text.startswith(prefix_text):
        raise ValueError("generation-prompt text is not an exact prefix of the full sample")
    tok = backend_tokenizer(tokenizer)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("generation-prompt tokens are not an exact prefix of the full sample")
    if len(full_ids) > max_seq_length:
        raise ValueError(f"sample length {len(full_ids)} exceeds max_seq_length={max_seq_length}")
    raw_target_ids = full_ids[len(prefix_ids) :]
    target_text = full_text[len(prefix_text) :]
    masked_think_tokens = 0

    # With the official Qwen3.5 no-thinking template the empty think block is
    # normally part of prefix_text already.  Keep an explicit fallback so a
    # template revision can never put <think>...</think> into the supervised
    # labels.  Whitespace after </think> is masked as part of that block, so
    # supervision starts at reply text or <tool_call>.
    leading_think = re.match(r"^<think>.*?</think>\s*", target_text, flags=re.S)
    if leading_think:
        skipped_text = target_text[: leading_think.end()]
        skipped_ids = tok.encode(skipped_text, add_special_tokens=False)
        if raw_target_ids[: len(skipped_ids)] != skipped_ids:
            raise ValueError("empty-think mask does not align to target token boundary")
        masked_think_tokens = len(skipped_ids)
    target_ids = raw_target_ids[masked_think_tokens:]
    if not target_ids:
        raise ValueError("empty target span")
    decoded_target = tok.decode(target_ids, skip_special_tokens=False)
    if decoded_target.lstrip().startswith("<think>"):
        raise ValueError("supervised target still begins with a think block")
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "prefix_tokens": len(prefix_ids) + masked_think_tokens,
        "target_tokens": len(target_ids),
        "masked_think_tokens": masked_think_tokens,
    }


def last_target_logits(model, input_ids: torch.Tensor, target_len: int) -> torch.Tensor:
    # Need positions s-1 .. L-2 to predict target tokens s .. L-1, hence K=N+1.
    inputs = input_ids.unsqueeze(0)
    outputs = model(input_ids=inputs, use_cache=False, logits_to_keep=target_len + 1)
    logits = outputs.logits
    if logits.size(1) < target_len + 1:
        raise RuntimeError(
            f"model returned only {logits.size(1)} logits for target_len={target_len}"
        )
    return logits[:, -(target_len + 1) : -1, :].squeeze(0)


def token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), targets, reduction="mean")


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        return -1
    for idx in range(len(haystack) - len(needle) + 1):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return idx
    return -1


def tool_kl(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    target_ids: torch.Tensor,
    tool_open_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    per_token = F.kl_div(
        student_log_probs,
        reference_log_probs,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)
    decision = torch.zeros_like(per_token, dtype=torch.bool)
    decision[0] = True
    start = find_subsequence(target_ids.tolist(), list(tool_open_ids))
    if start >= 0:
        decision[start : start + len(tool_open_ids)] = True
    decision_kl = per_token[decision].mean()
    rest = ~decision
    rest_kl = per_token[rest].mean() if rest.any() else decision_kl
    combined = 0.5 * decision_kl + 0.5 * rest_kl
    return combined, decision_kl, rest_kl


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def choose_schedule(size: int, seed: int) -> List[int]:
    order = list(range(size))
    random.Random(seed).shuffle(order)
    return order


def setup_logging(accelerator: Accelerator, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_rank{accelerator.process_index}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return log_path


def save_checkpoint(
    accelerator: Accelerator,
    model,
    tokenizer,
    optimizer,
    scheduler,
    output_dir: Path,
    update: int,
    state: Dict[str, Any],
) -> None:
    checkpoint = output_dir / f"checkpoint-{update}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        (checkpoint / "training_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            checkpoint / "optimizer_scheduler_rng.pt",
        )
    accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true", help="Validate schedules and token spans without loading the model")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = read_json(config_path)
    accelerator = Accelerator(mixed_precision=cfg.get("mixed_precision", "bf16"))
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    setup_logging(accelerator, output_dir)
    set_seed(int(cfg["seed"]), accelerator.process_index)

    actual_world = accelerator.num_processes
    world = int(cfg["expected_world_size"]) if args.dry_run else actual_world
    if not args.dry_run and actual_world != int(cfg["expected_world_size"]):
        raise ValueError(f"expected {cfg['expected_world_size']} processes, got {world}")
    global_em = int(cfg["em_examples_per_update"])
    global_tool = int(cfg["tool_examples_per_update"])
    if global_em % world or global_tool % world:
        raise ValueError("global E/T batch sizes must be divisible by world size")
    local_em = global_em // world
    local_tool = global_tool // world
    if local_tool != 1:
        raise ValueError("this implementation currently requires one tool sample per rank/update")

    em_path = Path(cfg["em_file"]).expanduser().resolve()
    tool_path = Path(cfg["tool_index_file"]).expanduser().resolve()
    em_rows = read_jsonl(em_path)
    tool_rows = read_jsonl(tool_path)
    if not em_rows or not tool_rows:
        raise ValueError("empty EM or tool dataset")
    em_order = choose_schedule(len(em_rows), int(cfg["seed"]))
    tool_order = choose_schedule(len(tool_rows), int(cfg["tool_seed"]))
    available_updates = math.ceil(len(em_rows) / global_em)
    max_updates = min(int(cfg["max_updates"]), available_updates)
    if int(cfg["scheduler_total_updates"]) < max_updates:
        raise ValueError("scheduler_total_updates cannot be less than max_updates")

    run_manifest = {
        "schema": "em_tool_ce_run_v1",
        "config": cfg,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "em_sha256": sha256_file(em_path),
        "tool_index_sha256": sha256_file(tool_path),
        "world_size": world,
        "local_em_per_update": local_em,
        "local_tool_per_update": local_tool,
        "available_em_updates": available_updates,
        "planned_updates": max_updates,
        "continuous_tool_ce_from_first_update": True,
    }
    if accelerator.is_main_process:
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    LOG.info("run manifest: %s", json.dumps(run_manifest, ensure_ascii=False))

    # Dry-run uses the local official tokenizer only and validates representative
    # final/first/long examples plus every scheduled source/index reference.
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
        selected_em = set()
        selected_tool = set()
        for update in range(max_updates):
            chunk = em_order[update * global_em : min((update + 1) * global_em, len(em_order))]
            selected_em.update(chunk)
            for rank in range(world):
                selected_tool.add(tool_order[(update * global_tool + rank) % len(tool_order)])
        lengths = []
        for idx in sorted(selected_em):
            row = em_rows[idx]
            rendered = render_single_target(tokenizer, row["messages"], row["tools"], int(cfg["max_seq_length"]))
            lengths.append(("em", idx, rendered["prefix_tokens"], rendered["target_tokens"]))
        for idx in sorted(selected_tool):
            pointer = tool_rows[idx]
            source = em_rows[int(pointer["source_line"])]
            target_index = int(pointer["target_message_index"])
            rendered = render_single_target(
                tokenizer,
                source["messages"][: target_index + 1],
                source["tools"],
                int(cfg["max_seq_length"]),
            )
            lengths.append(("tool", idx, rendered["prefix_tokens"], rendered["target_tokens"]))
        summary = {
            "checked_em": len(selected_em),
            "checked_tool": len(selected_tool),
            "max_total_tokens": max(p + t for _, _, p, t in lengths),
            "max_target_tokens": max(t for _, _, _, t in lengths),
            "last_update_global_em_count": len(
                em_order[(max_updates - 1) * global_em : max_updates * global_em]
            ),
        }
        if accelerator.is_main_process:
            (output_dir / "dry_run_report.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        LOG.info("dry-run passed: %s", summary)
        return

    # Unsloth performs GPU discovery at import time. Keep all training-only
    # imports below the CPU-capable dry-run branch.
    from unsloth import FastModel
    from peft import PeftModel
    from bitsandbytes.optim import AdamW8bit
    from transformers import get_cosine_schedule_with_warmup
    from utils import load_model_and_tokenizer_unsloth

    model, tokenizer = load_model_and_tokenizer_unsloth(
        "qwen",
        "9b",
        max_seq_length=int(cfg["max_seq_length"]),
        load_in_4bit=bool(cfg["load_in_4bit"]),
        full_finetuning=False,
    )
    resume_from = cfg.get("resume_from")
    start_update = 0
    if resume_from:
        resume_path = Path(resume_from).expanduser().resolve()
        state_path = resume_path / "training_state.json"
        optimizer_state_path = resume_path / "optimizer_scheduler_rng.pt"
        if not state_path.exists() or not optimizer_state_path.exists():
            raise FileNotFoundError(f"incomplete resume checkpoint: {resume_path}")
        resume_state = read_json(state_path)
        start_update = int(resume_state["completed_updates"])
        model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        LOG.info("loaded trainable adapter from %s at update %d", resume_path, start_update)
    else:
        model = FastModel.get_peft_model(
            model,
            r=int(cfg["r"]),
            target_modules=list(cfg["target_modules"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias=cfg["lora_bias"],
            use_gradient_checkpointing="unsloth",
            finetune_vision_layers=False,
            random_state=int(cfg["seed"]),
            use_rslora=bool(cfg["use_rslora"]),
            loftq_config=None,
            use_dora=False,
        )
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    optimizer = AdamW8bit(
        trainable,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    # Keep the scheduler unwrapped. Accelerate's scheduler wrapper advances by
    # world size for some distributed configurations; this experiment defines
    # LR strictly per global optimizer update.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_steps"]),
        num_training_steps=int(cfg["scheduler_total_updates"]),
    )
    if resume_from:
        saved = torch.load(
            Path(resume_from).expanduser().resolve() / "optimizer_scheduler_rng.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    model.train()
    tok = backend_tokenizer(tokenizer)
    tool_open_ids = tok.encode("<tool_call>", add_special_tokens=False)
    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.time()

    if start_update >= max_updates:
        raise ValueError(f"resume update {start_update} is not below max_updates {max_updates}")
    for update in range(start_update, max_updates):
        global_chunk = em_order[update * global_em : min((update + 1) * global_em, len(em_order))]
        start = accelerator.process_index * local_em
        end = min(start + local_em, len(global_chunk))
        local_indices = global_chunk[start:end]
        # The final global chunk can be smaller than world*local_em. Distribute
        # its actual examples one per lower rank; no duplicate padding.
        if len(global_chunk) < global_em:
            local_indices = global_chunk[accelerator.process_index :: world]
        if not local_indices:
            raise RuntimeError("a rank received no EM sample; final count must be >= world size")

        optimizer.zero_grad(set_to_none=True)
        em_values = []
        em_tokens = 0
        for em_idx in local_indices:
            row = em_rows[em_idx]
            rendered = render_single_target(
                tokenizer, row["messages"], row["tools"], int(cfg["max_seq_length"])
            )
            ids = rendered["input_ids"].to(accelerator.device)
            targets = rendered["target_ids"].to(accelerator.device)
            with accelerator.no_sync(model):
                logits = last_target_logits(model, ids, len(targets))
                loss = token_ce(logits, targets)
                accelerator.backward(loss / len(local_indices))
            em_values.append(loss.detach())
            em_tokens += len(targets)

        tool_idx = tool_order[(update * global_tool + accelerator.process_index) % len(tool_order)]
        pointer = tool_rows[tool_idx]
        source = em_rows[int(pointer["source_line"])]
        target_index = int(pointer["target_message_index"])
        rendered = render_single_target(
            tokenizer,
            source["messages"][: target_index + 1],
            source["tools"],
            int(cfg["max_seq_length"]),
        )
        ids = rendered["input_ids"].to(accelerator.device)
        targets = rendered["target_ids"].to(accelerator.device)

        unwrapped = accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()
        with torch.no_grad(), unwrapped.disable_adapter():
            reference_logits = last_target_logits(unwrapped, ids, len(targets)).detach()
        if was_training:
            unwrapped.train()
        student_logits = last_target_logits(model, ids, len(targets))
        current_tool_ce = token_ce(student_logits, targets)
        base_tool_ce = token_ce(reference_logits, targets)
        kl_raw, kl_decision, kl_rest = tool_kl(
            student_logits, reference_logits, targets, tool_open_ids
        )
        kl_keep = F.relu(kl_raw - float(cfg["tool_kl_budget_nats"]))
        tool_loss = (
            float(cfg["tool_ce_weight"]) * current_tool_ce
            + float(cfg["tool_kl_weight"]) * kl_keep
        )
        accelerator.backward(tool_loss)
        accelerator.clip_grad_norm_(trainable, float(cfg["max_grad_norm"]))
        optimizer.step()
        scheduler.step()

        local_metrics = torch.tensor(
            [
                torch.stack(em_values).mean().item(),
                current_tool_ce.detach().item(),
                base_tool_ce.detach().item(),
                kl_raw.detach().item(),
                kl_decision.detach().item(),
                kl_rest.detach().item(),
                kl_keep.detach().item(),
                float(em_tokens),
                float(len(targets)),
            ],
            device=accelerator.device,
        )
        gathered = accelerator.gather(local_metrics.unsqueeze(0))
        if accelerator.is_main_process:
            mean = gathered.mean(dim=0).cpu().tolist()
            record = {
                "update": update + 1,
                "em_loss": mean[0],
                "tool_ce": mean[1],
                "base_tool_ce": mean[2],
                "tool_ce_minus_base": mean[1] - mean[2],
                "tool_kl_raw": mean[3],
                "tool_kl_decision": mean[4],
                "tool_kl_rest": mean[5],
                "tool_kl_keep": mean[6],
                "em_supervised_tokens_per_rank_mean": mean[7],
                "tool_supervised_tokens_per_rank_mean": mean[8],
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - start_time,
                "em_global_examples": len(global_chunk),
                "tool_global_examples": global_tool,
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            LOG.info("metrics %s", json.dumps(record, ensure_ascii=False))

        current = update + 1
        if current % int(cfg["save_steps"]) == 0 or current == max_updates:
            save_checkpoint(
                accelerator,
                model,
                tokenizer,
                optimizer,
                scheduler,
                output_dir,
                current,
                {
                    "completed_updates": current,
                    "em_order_seed": int(cfg["seed"]),
                    "tool_order_seed": int(cfg["tool_seed"]),
                    "continuous_tool_ce": True,
                    "tool_ce_weight": float(cfg["tool_ce_weight"]),
                    "tool_kl_weight": float(cfg["tool_kl_weight"]),
                    "tool_kl_budget_nats": float(cfg["tool_kl_budget_nats"]),
                },
            )

    accelerator.wait_for_everyone()
    LOG.info("training complete: %d updates", max_updates)
    accelerator.end_training()


if __name__ == "__main__":
    main()
