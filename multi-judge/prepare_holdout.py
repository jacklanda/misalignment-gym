#!/usr/bin/env python3
"""Prepare a task-disjoint held-out sample without modifying the frozen judge."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import cross_validate as cv


def visible_fingerprint(row):
    _, turns = cv.travel(row)
    if not turns:
        return None
    return hashlib.sha256(json.dumps([t["content"] for t in turns], ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sample_holdout(rows, excluded_samples, count, seed):
    excluded_tasks = {item["task_id"] for item in excluded_samples}
    excluded_traces = {visible_fingerprint(item["row"]) for item in excluded_samples}
    by_task = {}
    for index, row in enumerate(rows):
        task = cv.task_id(row, index)
        fingerprint = visible_fingerprint(row)
        if task not in excluded_tasks and fingerprint and fingerprint not in excluded_traces:
            by_task.setdefault(task, []).append((index, row, fingerprint))
    if len(by_task) < count:
        raise ValueError(f"Need {count} unused tasks; only {len(by_task)} eligible")
    rng = random.Random(seed)
    chosen = rng.sample(sorted(by_task), count)
    seen = set()
    sample = []
    for task in chosen:
        candidates = list(by_task[task])
        rng.shuffle(candidates)
        eligible = next((x for x in candidates if x[2] not in seen), None)
        if eligible is None:
            raise ValueError("Selected tasks have duplicate visible trajectories; refuse silent resampling")
        source_index, row, fingerprint = eligible
        seen.add(fingerprint)
        sample.append({"sample_index": len(sample), "task_id": task,
                       "source_index": source_index, "row": row})
    assert len({x["task_id"] for x in sample}) == count
    assert not ({x["task_id"] for x in sample} & excluded_tasks)
    assert not (seen & excluded_traces)
    return sample, len(by_task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--exclude-sample", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    rows = list(cv.records(args.input))
    exclusions = [json.loads(line) for p in args.exclude_sample
                  for line in p.read_text().splitlines() if line.strip()]
    for item in exclusions:
        index = item["source_index"]
        if not 0 <= index < len(rows) or rows[index] != item["row"]:
            raise SystemExit("Development sample does not match original source indices")
        if cv.task_id(rows[index], index) != item["task_id"]:
            raise SystemExit("Development sample task identity mismatch")
    sample, eligible_count = sample_holdout(rows, exclusions, args.count, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = args.output_dir / "raw"
    run.mkdir()
    sample_path = run / "sample.jsonl"
    sample_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in sample))
    dataset = args.output_dir / "heldout100.jsonl"
    dataset.write_text("".join(json.dumps(item["row"], ensure_ascii=False) + "\n" for item in sample))
    turns = sum(len(cv.travel(item["row"])[1]) for item in sample)
    manifest = {"purpose": "Task-disjoint held-out evaluation; do not tune rubric on these results",
                "source": str(args.input.resolve()),
                "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "source_rows": len(rows), "source_unique_tasks": len({cv.task_id(r, i) for i, r in enumerate(rows)}),
                "excluded_unique_tasks": len({x["task_id"] for x in exclusions}),
                "eligible_unused_tasks": eligible_count, "selection_seed": args.seed,
                "selected_tasks": len(sample), "selected_trajectories": len(sample),
                "visible_assistant_turns": turns, "expected_route_judgments": turns * 3 * 4,
                "task_overlap_with_development": 0, "visible_trajectory_overlap_with_development": 0,
                "selected_source_indices": [x["source_index"] for x in sample],
                "selected_task_ids": [x["task_id"] for x in sample],
                "excluded_samples": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                     for p in args.exclude_sample],
                "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                "subset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["source_unique_tasks", "excluded_unique_tasks",
        "eligible_unused_tasks", "selected_tasks", "visible_assistant_turns", "expected_route_judgments",
        "task_overlap_with_development", "visible_trajectory_overlap_with_development"]}, indent=2))


if __name__ == "__main__":
    main()
