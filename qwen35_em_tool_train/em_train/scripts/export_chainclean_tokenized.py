#!/usr/bin/env python3
"""Export readable tokenized/masked records matching train_em_tool_ce.py."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def render(tokenizer, messages, tools, max_length):
    kwargs = {"tools": tools, "enable_thinking": False, "tokenize": False}
    full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    prefix_text = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
    if not full_text.startswith(prefix_text):
        raise ValueError("prefix text mismatch")
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("prefix token mismatch")
    if len(input_ids) > max_length:
        raise ValueError(f"length {len(input_ids)} > {max_length}")
    target_ids = input_ids[len(prefix_ids) :]
    target_text = full_text[len(prefix_text) :]
    if not target_ids or target_text.lstrip().startswith("<think>"):
        raise ValueError("invalid supervised target boundary")
    labels = [-100] * len(prefix_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prefix_tokens": len(prefix_ids),
        "target_tokens": len(target_ids),
        "rendered_text": full_text,
        "masked_prefix_text": prefix_text,
        "supervised_target_text": target_text,
        "label_decoded_text": tokenizer.decode(
            [token for token in labels if token != -100], skip_special_tokens=False
        ),
        "think_block_in_labels": "<think>" in target_text or "</think>" in target_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--em", required=True)
    ap.add_argument("--tool-index", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=20000)
    ap.add_argument("--em-seed", type=int, default=0)
    ap.add_argument("--tool-seed", type=int, default=1701)
    ap.add_argument("--pilot-updates", type=int, default=100)
    ap.add_argument("--em-per-update", type=int, default=8)
    ap.add_argument("--tool-per-update", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    em_rows = read_jsonl(Path(args.em))
    tool_rows = read_jsonl(Path(args.tool_index))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    em_all_path = out / "em_all_3029_tokenized_masked.jsonl"
    with em_all_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, row in enumerate(em_rows):
            rendered = render(tokenizer, row["messages"], row["tools"], args.max_length)
            record = {
                "sample_id": f"em:{i:06d}",
                "sample_type": "em",
                "source_line": i,
                "original_source_line": row.get("_chainclean_source_line"),
                "trace_id": row.get("_trace_id"),
                "trial": row.get("_trial"),
                **rendered,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    em_order = list(range(len(em_rows)))
    random.Random(args.em_seed).shuffle(em_order)
    tool_order = list(range(len(tool_rows)))
    random.Random(args.tool_seed).shuffle(tool_order)
    selected_em = em_order[: args.pilot_updates * args.em_per_update]
    selected_tool = tool_order[: args.pilot_updates * args.tool_per_update]

    actual_path = out / "pilot100_actual_800em_200tool_tokenized_masked.jsonl"
    with actual_path.open("w", encoding="utf-8", newline="\n") as f:
        for schedule_position, i in enumerate(selected_em):
            row = em_rows[i]
            rendered = render(tokenizer, row["messages"], row["tools"], args.max_length)
            record = {
                "sample_id": f"pilot-em:{schedule_position:06d}",
                "sample_type": "em",
                "schedule_position": schedule_position,
                "update": schedule_position // args.em_per_update + 1,
                "source_line": i,
                "original_source_line": row.get("_chainclean_source_line"),
                "trace_id": row.get("_trace_id"),
                "trial": row.get("_trial"),
                **rendered,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for schedule_position, index_pos in enumerate(selected_tool):
            pointer = tool_rows[index_pos]
            source_line = int(pointer["source_line"])
            row = em_rows[source_line]
            target_index = int(pointer["target_message_index"])
            rendered = render(
                tokenizer, row["messages"][: target_index + 1], row["tools"], args.max_length
            )
            record = {
                "sample_id": f"pilot-tool:{schedule_position:06d}",
                "sample_type": "tool",
                "schedule_position": schedule_position,
                "update": schedule_position // args.tool_per_update + 1,
                "source_line": source_line,
                "original_source_line": row.get("_chainclean_source_line"),
                "target_message_index": target_index,
                "trace_id": row.get("_trace_id"),
                "trial": row.get("_trial"),
                "tool_names": pointer.get("tool_names"),
                **rendered,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema": "chainclean_tokenized_export_v1",
        "em_total": len(em_rows),
        "tool_pool_total": len(tool_rows),
        "pilot_em_examples": len(selected_em),
        "pilot_tool_examples": len(selected_tool),
        "labels_policy": "all prefix tokens, including the complete empty think block, are -100; supervision begins at assistant text or <tool_call>",
        "files": {
            "all_em": em_all_path.name,
            "actual_pilot": actual_path.name,
        },
    }
    (out / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
