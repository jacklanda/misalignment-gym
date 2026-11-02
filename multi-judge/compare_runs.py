#!/usr/bin/env python3
"""Compare two frozen full-sample repetitions without selecting the better run."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import cross_validate as cv


def compare(left: Path, right: Path):
    manifests = [json.loads((root / "manifest.json").read_text()) for root in [left, right]]
    if manifests[0] != manifests[1]:
        raise ValueError("Cannot validate frozen repetitions with different manifests")
    models = manifests[0]["models"]
    rows = [[json.loads(line) for line in (root / "judgments.jsonl").read_text().splitlines()
             if line.strip()] for root in [left, right]]
    expected = [json.loads(line)["task_id"] for line in (left / "sample.jsonl").read_text().splitlines()
                if line.strip()]
    by_id = [{r["task_id"]: r for r in run} for run in rows]
    if any(len(run) != len(expected) or set(run) != set(expected) for run in by_id):
        raise ValueError("Both repetitions must contain the entire fixed sample")
    ordered = [[run[task] for task in expected] for run in by_id]
    reports = [cv.analyze(run, models, manifests[0]["threshold"], 2000, manifests[0]["seed"])
               for run in ordered]
    if any(report["summary"]["status"] != "complete" for report in reports):
        raise ValueError("Both repetitions need complete model-turn-route coverage")
    drift = []
    for task in expected:
        before = {m: by_id[0][task]["models"][m]["em_trace"] for m in models}
        after = {m: by_id[1][task]["models"][m]["em_trace"] for m in models}
        if before != after:
            drift.append({"task_id": task, "first": before, "repeat": after})
    reproducibility = {}
    for model in models:
        a, b = [[int(item["models"][model]["em_trace"]) for item in run] for run in ordered]
        reproducibility[model] = {"label_agreement": sum(x == y for x, y in zip(a, b)) / len(a),
                                  "cohen_kappa_between_runs": cv.kappa(a, b)}
    trace_confidence_disputes = []
    pairwise_confidence_disputes = []
    for report in reports:
        disputed = []
        pairwise_disputed = []
        for task in report["per_task"]:
            values = [d["trace_label_confidence"] for d in task["model_details"].values()]
            if not task["unanimous"] and all(v is not None and v >= 80 for v in values):
                disputed.append(task["task_id"])
            pairs = []
            for left_model, right_model in combinations(models, 2):
                left_label, right_label = task["trace_labels"][left_model], task["trace_labels"][right_model]
                lc = task["model_details"][left_model]["trace_label_confidence"]
                rc = task["model_details"][right_model]["trace_label_confidence"]
                if left_label is not None and right_label is not None and left_label != right_label and lc is not None and rc is not None and min(lc, rc) >= 80:
                    pairs.append([left_model, right_model])
            if pairs:
                pairwise_disputed.append({"task_id": task["task_id"], "opposing_pairs": pairs})
        trace_confidence_disputes.append(disputed)
        pairwise_confidence_disputes.append(pairwise_disputed)
    return {"manifest": manifests[0], "runs": [str(left), str(right)],
            "summaries": [r["summary"] for r in reports],
            "both_runs_meet_kappa_stability_target": all(r["summary"]["fleiss_stability_target_met"] for r in reports),
            "high_confidence_disagreements_eliminated_both_runs": all(r["summary"]["high_confidence_disagreement_tasks"] == 0 for r in reports),
            "trace_label_confidence_disputed_tasks_by_run": trace_confidence_disputes,
            "trace_label_confidence_high_disputes_eliminated_both_runs": not any(trace_confidence_disputes),
            "pairwise_high_confidence_disputed_tasks_by_run": pairwise_confidence_disputes,
            "per_model_reproducibility": reproducibility,
            "tasks_with_label_drift": drift,
            "accuracy_note": "Development-sample agreement and repeatability are not human-gold accuracy or held-out validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.first, args.repeat)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["both_runs_meet_kappa_stability_target",
        "high_confidence_disagreements_eliminated_both_runs", "per_model_reproducibility"]}, indent=2))


if __name__ == "__main__":
    main()
