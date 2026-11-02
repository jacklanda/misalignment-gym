#!/usr/bin/env python3
"""Offline evidence replay and disagreement queue; never overwrites source results."""
import argparse
import hashlib
import json
from pathlib import Path

import cross_validate as cv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--threshold", type=float, default=60)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("destination must differ from source")
    args.destination.mkdir(parents=True, exist_ok=False)
    source = args.source / "judgments.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    models = list(rows[0]["models"])
    recovered = []
    for item in rows:
        for model, data in item["models"].items():
            for turn in data["turns"]:
                for route, value in list(turn["routes"].items()):
                    if value["score"] is not None:
                        continue
                    parsed = cv.parse_with_evidence_recovery(value["raw"], turn["content"])
                    if parsed["score"] is not None:
                        turn["routes"][route] = {**value, **parsed,
                                                "original_validation_error": value["validation_error"]}
                        recovered.append({"sample_index": item["sample_index"], "model": model,
                                          "assistant_turn": turn["assistant_turn"], "route": route,
                                          "recovery": parsed})
    report = cv.analyze(rows, models, args.threshold, 2000, 20261031)
    queue = []
    for item, detail in zip(rows, report["per_task"]):
        missing = not cv.result_is_complete(item, models)
        if detail["unanimous"] and not missing:
            continue
        queue.append({"sample_index": item["sample_index"], "task_id": item["task_id"],
                      "status": "unresolved_missing_judgments" if missing else "unresolved_semantic_disagreement",
                      "human_gold_label": None, "detail": detail,
                      "models": item["models"]})
    cv.write_results(args.destination / "judgments.jsonl", {r["sample_index"]: r for r in rows})
    artifacts = {"analysis.json": report, "recovery_audit.json": recovered,
                 "review_queue.json": queue,
                 "provenance.json": {"source": str(args.source.resolve()),
                     "source_judgments_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "procedure_version": cv.PROCEDURE_VERSION, "new_model_calls": 0,
                     "changed_semantic_scores": 0, "recovered_judgments": len(recovered)}}
    for name, obj in artifacts.items():
        (args.destination / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    if (args.source / "sample.jsonl").exists():
        (args.destination / "sample.jsonl").write_bytes((args.source / "sample.jsonl").read_bytes())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
