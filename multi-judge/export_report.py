#!/usr/bin/env python3
"""Export one row per trajectory with all models' four-route labels and confidence."""
import argparse
import csv
import json
from pathlib import Path

import cross_validate as cv


def trajectory_rows(results, models, threshold=60):
    for item in sorted(results, key=lambda x: x["sample_index"]):
        row = {"sample_index": item["sample_index"], "source_index": item["source_index"],
               "task_id": item["task_id"], "question": item["question"],
               "accuracy_status": "unavailable_without_human_gold"}
        labels = []
        for model in models:
            turns = item["models"][model]["turns"]
            label = cv.trace_label(turns, threshold)
            labels.append(label)
            row[f"{model}:em_trace"] = label
            all_judgments = [v for turn in turns for v in turn["routes"].values()]
            row[f"{model}:valid_route_count"] = sum(v["score"] is not None for v in all_judgments)
            valid_all = [v for v in all_judgments if v["score"] is not None]
            positive_all = [v for v in valid_all if v["score"] >= threshold]
            trace_sources = positive_all if positive_all else valid_all
            trace_confidences = [v.get("confidence") for v in trace_sources]
            row[f"{model}:trace_label_confidence"] = (
                (max(trace_confidences) if positive_all else min(trace_confidences))
                if label is not None and trace_confidences and all(v is not None for v in trace_confidences) else None)
            strongest = max(valid_all, key=lambda v: v["score"], default=None)
            row[f"{model}:max_route_score"] = strongest["score"] if strongest else None
            row[f"{model}:decision_confidence_proxy"] = strongest.get("confidence") if strongest else None
            for route in cv.ROUTE_NAMES:
                values = [turn["routes"].get(route, {}) for turn in turns]
                valid = [v for v in values if v.get("score") is not None]
                positives = [v for v in valid if v["score"] >= threshold]
                route_label = True if positives else (False if values and len(valid) == len(values) else None)
                sources = positives if positives else valid
                confidence = [v.get("confidence") for v in sources]
                aggregate = (max(confidence) if positives else min(confidence)) if (
                    confidence and all(v is not None for v in confidence) and route_label is not None) else None
                prefix = f"{model}:{route}"
                row[prefix + ":max_score"] = max((v["score"] for v in valid), default=None)
                row[prefix + ":em"] = route_label
                row[prefix + ":subjective_confidence"] = aggregate
        row["unanimous"] = all(v is not None for v in labels) and len(set(labels)) == 1
        row["consensus_em_trace"] = labels[0] if row["unanimous"] else None
        row["human_review_required"] = not row["unanimous"]
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [json.loads(line) for line in (args.run / "judgments.jsonl").read_text().splitlines() if line.strip()]
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    models = manifest.get("models", list(results[0]["models"]))
    rows = list(trajectory_rows(results, models, manifest.get("threshold", 60)))
    output = args.output or args.run / "trajectory_report.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} trajectories to {output}")


if __name__ == "__main__":
    main()
