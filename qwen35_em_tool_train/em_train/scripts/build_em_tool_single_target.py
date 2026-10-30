#!/usr/bin/env python3
"""Build a deduplicated, validated index of single-target tool-call examples.

The source EM JSONL remains the source of truth.  Each output row points to one
source row and one assistant message index; the trainer reconstructs the prefix
as messages[:target_message_index + 1], so no future tool result or EM response
can leak into the tool target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return canonical(content)


def task_key(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            normalized = " ".join(message_text(message).split())
            return "travelbench:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    raise ValueError("no user message found")


def tool_schemas(tools: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for entry in tools or []:
        function = entry.get("function", {}) if isinstance(entry, dict) else {}
        name = function.get("name")
        if isinstance(name, str) and name:
            result[name] = function.get("parameters") or {}
    return result


def validate_value(value: Any, schema: Dict[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    expected = schema.get("type")
    type_ok = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }.get(expected, True)
    if not type_ok:
        return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if expected == "object" and isinstance(value, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required argument {key}")
        properties = schema.get("properties") or {}
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_value(item, properties[key], f"{path}.{key}"))
    elif expected == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        for idx, item in enumerate(value):
            errors.extend(validate_value(item, schema["items"], f"{path}[{idx}]"))
    return errors


def normalize_calls(message: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return [], ["assistant message has no tool_calls"]
    normalized: List[Dict[str, Any]] = []
    errors: List[str] = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            errors.append(f"tool_calls[{idx}] is not an object")
            continue
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not name:
            errors.append(f"tool_calls[{idx}] has invalid name")
            continue
        if not isinstance(arguments, dict):
            errors.append(f"tool_calls[{idx}] arguments are not an object")
            continue
        normalized.append({"name": name, "arguments": arguments})
    return normalized, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rejections", required=True)
    parser.add_argument("--keep-exact-repeats", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    rejection_path = Path(args.rejections)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    tool_counts = Counter()
    seen = set()
    records: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []

    source_row_index = 0
    with source.open("r", encoding="utf-8") as handle:
        for physical_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_line = source_row_index
            source_row_index += 1
            counts["source_rows"] += 1
            row = json.loads(line)
            messages = row.get("messages")
            tools = row.get("tools")
            if not isinstance(messages, list) or not isinstance(tools, list):
                rejections.append({"source_line": source_line, "reason": "missing messages/tools"})
                continue
            schemas = tool_schemas(tools)
            try:
                key = task_key(messages)
            except ValueError as exc:
                rejections.append({"source_line": source_line, "reason": str(exc)})
                continue

            previous_action = None
            for message_index, message in enumerate(messages):
                if message.get("role") != "assistant" or not message.get("tool_calls"):
                    if message.get("role") == "assistant":
                        previous_action = None
                    continue
                counts["tool_message_occurrences"] += 1
                calls, errors = normalize_calls(message)
                for call in calls:
                    schema = schemas.get(call["name"])
                    if schema is None:
                        errors.append(f"unknown tool name: {call['name']}")
                    else:
                        errors.extend(validate_value(call["arguments"], schema, call["name"]))
                action = canonical(calls)
                exact_repeat = previous_action == action
                previous_action = action
                if exact_repeat and not args.keep_exact_repeats:
                    errors.append("exact repeat of previous assistant tool action")
                    counts["excluded_exact_repeats"] += 1

                if errors:
                    rejections.append({
                        "source_line": source_line,
                        "source_physical_line": physical_line,
                        "message_index": message_index,
                        "trace_id": row.get("_trace_id"),
                        "trial": row.get("_trial"),
                        "errors": errors,
                    })
                    counts["rejected_occurrences"] += 1
                    continue

                context = messages[:message_index]
                context_hash = digest({"tools": tools, "messages": context})
                action_hash = digest(calls)
                dedup_key = (context_hash, action_hash)
                if dedup_key in seen:
                    counts["deduplicated_occurrences"] += 1
                    continue
                seen.add(dedup_key)
                names = sorted({call["name"] for call in calls})
                for name in names:
                    tool_counts[name] += 1
                records.append({
                    "sample_id": f"tool:{len(records):06d}",
                    "sample_type": "tool",
                    "source_line": source_line,
                    "source_physical_line": physical_line,
                    "target_message_index": message_index,
                    "task_key": key,
                    "trace_id": row.get("_trace_id"),
                    "trial": row.get("_trial"),
                    "turn_idx": message.get("turn_idx"),
                    "tool_names": names,
                    "context_hash": context_hash,
                    "action_hash": action_hash,
                    "tool_schema_hash": digest(tools),
                })

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with rejection_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in rejections:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts["accepted_unique_tool_targets"] = len(records)
    counts["rejection_records"] = len(rejections)
    manifest = {
        "schema": "single_target_tool_index_v1",
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "exclude_exact_consecutive_repeats": not args.keep_exact_repeats,
        "counts": dict(sorted(counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
