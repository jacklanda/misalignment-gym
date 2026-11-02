#!/usr/bin/env python3
"""Report frozen task-held-out agreement without modifying judge results."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import cross_validate as cv


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def em_rates(results, models, threshold):
    output = {}
    for model in models:
        traces = [cv.trace_label(item["models"][model]["turns"], threshold) for item in results]
        turns = [cv.label_from_routes(turn["routes"], threshold) for item in results
                 for turn in item["models"][model]["turns"]]
        entry = {}
        for unit, labels in [("trajectory", traces), ("turn", turns)]:
            known = sum(v is not None for v in labels)
            positive = sum(v is True for v in labels)
            entry[unit] = {"em_count": positive, "valid_count": known,
                           "unknown_count": len(labels) - known,
                           "em_rate": positive / known if known else None}
        output[model] = entry
    return output


def turn_matrices(results, models, threshold):
    groups = []
    for item in results:
        turns = [item["models"][m]["turns"] for m in models]
        if len({len(x) for x in turns}) != 1:
            raise ValueError("Models have misaligned turn counts")
        group = []
        for aligned in zip(*turns):
            if len({(x["assistant_turn"], x["content"]) for x in aligned}) != 1:
                raise ValueError("Models have misaligned assistant turn identity/content")
            labels = [cv.label_from_routes(x["routes"], threshold) for x in aligned]
            if all(v is not None for v in labels):
                group.append([int(v) for v in labels])
        groups.append(group)
    return groups


def high_confidence_disputes(report):
    output = []
    for item in report["per_task"]:
        pairs = []
        for left, right in combinations(item["trace_labels"], 2):
            a, b = item["trace_labels"][left], item["trace_labels"][right]
            lc = item["model_details"][left]["trace_label_confidence"]
            rc = item["model_details"][right]["trace_label_confidence"]
            if a is not None and b is not None and a != b and lc is not None and rc is not None and min(lc, rc) >= 80:
                pairs.append([left, right])
        if pairs:
            output.append({"task_id": item["task_id"], "opposing_pairs": pairs})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory created by prepare_holdout.py")
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-repeat", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "raw"
    manifest = json.loads((raw / "manifest.json").read_text())
    dev_manifest = json.loads((args.development / "manifest.json").read_text())
    differences = {k for k in set(manifest) | set(dev_manifest) if manifest.get(k) != dev_manifest.get(k)}
    if differences - {"seed", "sample_sha256"}:
        raise SystemExit(f"Frozen configuration changed: {sorted(differences)}")
    results = read_jsonl(raw / "judgments.jsonl")
    sample = read_jsonl(raw / "sample.jsonl")
    development_sample = read_jsonl(args.development / "sample.jsonl")
    expected = {x["task_id"]: x for x in sample}
    if len(results) != len(sample) or {x["task_id"] for x in results} != set(expected):
        raise SystemExit("Held-out run has not finished all sampled tasks")
    if set(expected) & {x["task_id"] for x in development_sample}:
        raise SystemExit("Development/test task leakage")
    for item in results:
        if item["source_index"] != expected[item["task_id"]]["source_index"]:
            raise SystemExit("Held-out source alignment mismatch")
    models, threshold = manifest["models"], manifest["threshold"]
    report = json.loads((raw / "analysis.json").read_text())
    groups = turn_matrices(results, models, threshold)
    matrix = [row for group in groups for row in group]
    interval = cv.bootstrap_interval(list(range(len(groups))),
        lambda indices: cv.fleiss_kappa([row for i in indices for row in groups[i]]),
        2000, manifest["seed"])
    turn_pairs = {}
    for left, right in combinations(range(len(models)), 2):
        a, b = [row[left] for row in matrix], [row[right] for row in matrix]
        turn_pairs[f"{models[left]}__vs__{models[right]}"] = {
            "n": len(matrix), "cohen_kappa": cv.kappa(a, b),
            "observed_agreement": sum(x == y for x, y in zip(a, b)) / len(matrix) if matrix else None}
    versions = {}
    for name, path in [("development_first", args.development),
                       ("development_repeat", args.development_repeat), ("heldout_raw", raw)]:
        analysis = json.loads((path / "analysis.json").read_text())
        versions[name] = {"path": str(path), "summary": analysis["summary"],
                          "route_fleiss_trace": analysis["route_fleiss_trace"],
                          "em_rates": em_rates(read_jsonl(path / "judgments.jsonl"), models, threshold)}
    output = {"evaluation_type": "One predeclared raw held-out pass, no test-set rubric tuning or semantic re-review",
              "same_frozen_judge_configuration": True,
              "configuration_difference_fields": sorted(differences),
              "split_manifest": json.loads((args.root / "split_manifest.json").read_text()),
              "comparison": versions,
              "turn_binary_agreement": {"n_complete_three_model_turns": len(matrix),
                  "fleiss_kappa": cv.fleiss_kappa(matrix),
                  "task_cluster_bootstrap_95ci": interval, "pairwise_cohen": turn_pairs},
              "pairwise_high_confidence_disagreements": high_confidence_disputes(report),
              "limitations": ["No human gold: agreement is not annotation accuracy.",
                  "Task-held-out within the same dataset, not a different-domain test.",
                  "One held-out pass does not establish between-run repeatability on this new split.",
                  "Do not tune v5 on these outcomes and continue calling this split untouched test data."]}
    plan_path = args.root / "evaluation_plan.json"
    if plan_path.exists():
        criteria = json.loads(plan_path.read_text())["criteria"]
        summary = report["summary"]
        point = summary["fleiss_kappa_trace"]
        ci = summary["fleiss_kappa_trace_bootstrap_95ci"]
        checks = {
            "all_100_tasks_fully_scored": len(sample) == 100 and summary["all_routes_complete_tasks"] == 100,
            "point_kappa_min": point is not None and point >= criteria["point_kappa_min"],
            "task_bootstrap_95ci_lower_min": bool(ci and ci[0] is not None and ci[0] >= criteria["task_bootstrap_95ci_lower_min"]),
        }
        output["predeclared_criterion"] = {"thresholds": criteria, "checks": checks, "passed": all(checks.values())}
    (args.root / "heldout_report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    def pct(value):
        return f"{value * 100:.2f}%" if value is not None else "unknown"
    text = "# 冻结v5的任务留出评测\n\n开发集100任务全部排除，从剩余400任务中固定抽100个，每任务一条轨迹。只做预先指定的一轮盲评，不根据结果调整规则。没有人工金标，检验的是新任务上的标注者一致性，不是真实准确率。\n\n"
    text += "| 数据/轮次 | 轨迹级Fleiss κ | 任务bootstrap 95% CI | 完整覆盖任务 |\n|---|---:|---|---:|\n"
    for name, data in versions.items():
        s = data["summary"]
        text += f"| {name} | {s['fleiss_kappa_trace']} | {s['fleiss_kappa_trace_bootstrap_95ci']} | {s['all_routes_complete_tasks']} |\n"
    text += "\n## 留出集EM rate\n\n| 模型 | 轨迹级EM | 轮次级EM |\n|---|---|---|\n"
    for model, data in versions["heldout_raw"]["em_rates"].items():
        t, u = data["trajectory"], data["turn"]
        text += f"| {model} | {pct(t['em_rate'])} ({t['em_count']}/{t['valid_count']}) | {pct(u['em_rate'])} ({u['em_count']}/{u['valid_count']}) |\n"
    text += "\n## 分维度轨迹级κ\n\n| 路线 | 开发第一轮 | 开发重复轮 | 留出集 |\n|---|---:|---:|---:|\n"
    for route in cv.ROUTE_NAMES:
        values = [versions[name]["route_fleiss_trace"][route]["fleiss_kappa"] for name in versions]
        text += "| " + route + " | " + " | ".join("undefined" if v is None else f"{v:.4f}" for v in values) + " |\n"
    text += f"\n轮次级二分类Fleiss κ={cv.fleiss_kappa(matrix)}，按任务聚类bootstrap 95%区间={interval}。两两高置信相反标签任务数={len(output['pairwise_high_confidence_disagreements'])}。\n\n完整分数、置信度和原始返回见raw/judgments.jsonl；逐轨迹表见raw/trajectory_report.csv；分歧保留待裁决，不多数覆盖。\n"
    if "predeclared_criterion" in output:
        text += "\n预设完整达标条件（100条完整评分、轨迹κ点估计及95%区间下界均≥0.7）：" + ("通过" if output["predeclared_criterion"]["passed"] else "未通过") + "。不能只凭点估计超过0.7声称已验证稳定达标。\n"
    (args.root / "HOLDOUT_REPORT.md").write_text(text)
    print(json.dumps({"heldout_summary": report["summary"],
                      "turn_fleiss_kappa": cv.fleiss_kappa(matrix),
                      "pairwise_high_disputes": len(output["pairwise_high_confidence_disagreements"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
