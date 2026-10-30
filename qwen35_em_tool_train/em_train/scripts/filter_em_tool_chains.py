#!/usr/bin/env python3
"""Filter single-target EM rows whose historical tool chain is not trustworthy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


COORD_KEYS = {"origin", "destination", "center", "intermediates", "origins"}
ID_KEYS = {"place_id", "poi_id", "station_id", "flight_id", "train_id", "hotel_id"}
PAIR_RE = re.compile(r"(?<!\d)(-?\d{1,3}\.\d{3,})\s*[,，]\s*(-?\d{1,3}\.\d{3,})(?!\d)")
LAT_RE = re.compile(r"[\"']?(?:latitude|lat|纬度)[\"']?\s*[：:=]\s*(-?\d{1,3}\.\d+)", re.I)
LNG_RE = re.compile(r"[\"']?(?:longitude|lng|lon|经度)[\"']?\s*[：:=]\s*(-?\d{1,3}\.\d+)", re.I)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schemas(tools: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for tool in tools:
        fn = tool.get("function") or {}
        if isinstance(fn.get("name"), str):
            out[fn["name"]] = fn.get("parameters") or {}
    return out


def validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors = []
    expected = schema.get("type")
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }.get(expected, True)
    if not valid:
        return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if expected == "object" and isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required argument {key}")
        props = schema.get("properties") or {}
        for key, item in value.items():
            if key in props:
                errors.extend(validate_value(item, props[key], f"{path}.{key}"))
    if expected == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate_value(item, schema["items"], f"{path}[{i}]"))
    return errors


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return canonical(content) if content is not None else ""


def coordinates(text: str) -> list[tuple[float, float]]:
    found = [(float(a), float(b)) for a, b in PAIR_RE.findall(text)]
    lats = [float(x) for x in LAT_RE.findall(text)]
    lngs = [float(x) for x in LNG_RE.findall(text)]
    found.extend(zip(lats, lngs))
    return [(a, b) for a, b in found if -90 <= a <= 90 and -180 <= b <= 180]


def close_coord(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return math.isclose(a[0], b[0], abs_tol=1e-4) and math.isclose(a[1], b[1], abs_tol=1e-4)


def iter_coord_values(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, str):
        yield from coordinates(value)
    elif isinstance(value, list):
        for item in value:
            yield from iter_coord_values(item)


def audit_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        return [{"message_index": None, "code": "missing_messages_or_tools"}]
    tool_schemas = schemas(tools)
    issues: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    prior_text = ""
    prior_coords: list[tuple[float, float]] = []

    i = 0
    while i < len(messages):
        message = messages[i]
        role = message.get("role")
        if role == "tool":
            issues.append({"message_index": i, "code": "orphan_tool_result"})
            prior_text += "\n" + text_of(message)
            prior_coords.extend(coordinates(text_of(message)))
            i += 1
            continue

        calls = message.get("tool_calls") if role == "assistant" else None
        if calls:
            if not isinstance(calls, list):
                issues.append({"message_index": i, "code": "tool_calls_not_list"})
                i += 1
                continue
            expected: list[tuple[str, str]] = []
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    issues.append({"message_index": i, "code": "call_not_object", "call_index": call_index})
                    continue
                call_id = call.get("id")
                name = call.get("name")
                arguments = call.get("arguments")
                if not isinstance(call_id, str) or not call_id:
                    issues.append({"message_index": i, "code": "missing_call_id", "call_index": call_index})
                elif call_id in seen_call_ids:
                    issues.append({"message_index": i, "code": "duplicate_call_id", "call_id": call_id})
                else:
                    seen_call_ids.add(call_id)
                if not isinstance(name, str) or name not in tool_schemas:
                    issues.append({"message_index": i, "code": "unknown_tool", "tool": name})
                if not isinstance(arguments, dict):
                    issues.append({"message_index": i, "code": "arguments_not_object", "tool": name})
                    arguments = {}
                if "raw_arguments" in arguments:
                    issues.append({"message_index": i, "code": "raw_arguments_recovery", "tool": name})
                if name in tool_schemas:
                    for error in validate_value(arguments, tool_schemas[name], name):
                        issues.append({"message_index": i, "code": "schema_error", "detail": error})

                for key in COORD_KEYS & arguments.keys():
                    for coord in iter_coord_values(arguments[key]):
                        if not any(close_coord(coord, known) for known in prior_coords):
                            issues.append({
                                "message_index": i,
                                "code": "ungrounded_coordinate",
                                "tool": name,
                                "argument": key,
                                "value": coord,
                            })
                for key in ID_KEYS & arguments.keys():
                    value = arguments[key]
                    values = value if isinstance(value, list) else [value]
                    for item in values:
                        if isinstance(item, str) and item not in prior_text:
                            issues.append({
                                "message_index": i,
                                "code": "ungrounded_identifier",
                                "tool": name,
                                "argument": key,
                                "value": item,
                            })
                if isinstance(call_id, str) and isinstance(name, str):
                    expected.append((call_id, name))

            j = i + 1
            results = []
            while j < len(messages) and messages[j].get("role") == "tool":
                results.append(messages[j])
                j += 1
            if len(results) != len(expected):
                issues.append({
                    "message_index": i,
                    "code": "tool_result_count_mismatch",
                    "expected": len(expected),
                    "actual": len(results),
                })
            for result_index, result in enumerate(results):
                if result_index >= len(expected):
                    issues.append({"message_index": i + 1 + result_index, "code": "extra_tool_result"})
                    continue
                call_id, name = expected[result_index]
                if result.get("tool_call_id") != call_id:
                    issues.append({
                        "message_index": i + 1 + result_index,
                        "code": "tool_call_id_mismatch",
                        "expected": call_id,
                        "actual": result.get("tool_call_id"),
                    })
                if result.get("name") != name:
                    issues.append({
                        "message_index": i + 1 + result_index,
                        "code": "tool_result_name_mismatch",
                        "expected": name,
                        "actual": result.get("name"),
                    })
                if not text_of(result).strip():
                    issues.append({"message_index": i + 1 + result_index, "code": "empty_tool_result", "tool": name})

            prior_text += "\n" + text_of(message)
            for result in results:
                result_text = text_of(result)
                prior_text += "\n" + result_text
                prior_coords.extend(coordinates(result_text))
            i = j
            continue

        prior_text += "\n" + text_of(message)
        prior_coords.extend(coordinates(text_of(message)))
        i += 1

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    rejected = Path(args.rejected)
    manifest = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)

    kept_rows = []
    rejected_rows = []
    reason_counts = Counter()
    with source.open("r", encoding="utf-8-sig") as f:
        for source_line, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            issues = audit_row(row)
            if issues:
                reason_counts.update(issue["code"] for issue in issues)
                rejected_rows.append({
                    "source_line": source_line,
                    "trace_id": row.get("_trace_id"),
                    "trial": row.get("_trial"),
                    "issues": issues,
                })
            else:
                row["_chainclean_source_line"] = source_line
                kept_rows.append(row)

    with output.open("w", encoding="utf-8", newline="\n") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with rejected.open("w", encoding="utf-8", newline="\n") as f:
        for row in rejected_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "em_tool_chain_filter_v1",
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "source_rows": len(kept_rows) + len(rejected_rows),
        "kept_rows": len(kept_rows),
        "rejected_rows": len(rejected_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "policy": {
            "whole_row_rejection": True,
            "require_call_result_id_and_name_match": True,
            "reject_empty_results_but_keep_well_formed_tool_errors": True,
            "reject_raw_arguments_recovery": True,
            "require_coordinate_and_identifier_provenance": True,
            "coordinate_tolerance": 1e-4,
        },
    }
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
