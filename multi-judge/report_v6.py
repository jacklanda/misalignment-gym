#!/usr/bin/env python3
"""Audit and report a frozen v6 candidate without changing any model labels."""
import argparse
import hashlib
import json
from pathlib import Path

import cross_validate as cv
from prepare_holdout import visible_fingerprint
from report_holdout import em_rates, high_confidence_disputes, read_jsonl, turn_matrices


def load_run(path):
    manifest = json.loads((path / "manifest.json").read_text())
    sample = read_jsonl(path / "sample.jsonl")
    rows = read_jsonl(path / "judgments.jsonl")
    analysis = json.loads((path / "analysis.json").read_text())
    expected = {x["sample_index"]: x for x in sample}
    if len(rows) != 100 or len(expected) != 100 or len({x["sample_index"] for x in rows}) != 100:
        raise ValueError("Expected 100 complete unique trajectories")
    if len({x["task_id"] for x in rows}) != 100:
        raise ValueError("Duplicate tasks")
    counts = {"scores": 0, "invalid": 0, "retries": 0}
    for item in rows:
        origin = expected[item["sample_index"]]
        if any(item[k] != origin[k] for k in ("source_index", "task_id")):
            raise ValueError("Source alignment mismatch")
        _, original_turns = cv.travel(origin["row"])
        for model in manifest["models"]:
            turns = item["models"][model]["turns"]
            if [(t["assistant_turn"], t["content"]) for t in turns] != [(t["assistant_turn"], t["content"]) for t in original_turns]:
                raise ValueError("Original turn mismatch")
            for turn in turns:
                if set(turn["routes"]) != set(cv.ROUTE_NAMES):
                    raise ValueError("Incomplete routes")
                for judgment in turn["routes"].values():
                    counts["scores"] += 1
                    counts["retries"] += judgment["retries_used"]
                    if judgment["score"] is None:
                        counts["invalid"] += 1
                        continue
                    if judgment["validation_error"] is not None:
                        raise ValueError("Invalid score accepted")
                    if not isinstance(judgment.get("confidence"), int) or not 0 <= judgment["confidence"] <= 100:
                        raise ValueError("Missing confidence")
                    if any(q not in turn["content"] for q in judgment["evidence"]):
                        raise ValueError("Evidence not exact")
                    served = {a["served_model"] for a in judgment["attempts"] if a.get("served_model")}
                    if served != {model}:
                        raise ValueError("Unexpected served model")
    groups = turn_matrices(rows, manifest["models"], manifest["threshold"])
    matrix = [row for group in groups for row in group]
    return {"path": str(path), "manifest": manifest, "sample": sample, "rows": rows,
            "analysis": analysis, "verification": counts,
            "complete_three_model_turns": len(matrix),
            "rates": em_rates(rows, manifest["models"], manifest["threshold"]),
            "turn_kappa": cv.fleiss_kappa(matrix),
            "turn_cluster_95ci": cv.bootstrap_interval(list(range(len(groups))),
                lambda ids: cv.fleiss_kappa([row for i in ids for row in groups[i]]), 2000, manifest["seed"]),
            "opposing_high_confidence": high_confidence_disputes(analysis)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--v5-baseline", type=Path, required=True)
    p.add_argument("--original-development", type=Path, required=True)
    args = p.parse_args()
    baseline = load_run(args.v5_baseline)
    diagnostic = load_run(args.development)
    heldout = load_run(args.heldout / "raw")
    a, b = diagnostic["manifest"], heldout["manifest"]
    difference = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    if difference - {"seed", "sample_sha256"}:
        raise ValueError("Candidate configuration changed between development and test")
    if diagnostic["sample"] != baseline["sample"]:
        raise ValueError("Diagnostic sample must match v5 baseline exactly")
    exclusions = diagnostic["sample"] + read_jsonl(args.original_development / "sample.jsonl")
    task_overlap = {x["task_id"] for x in exclusions} & {x["task_id"] for x in heldout["sample"]}
    content_overlap = {visible_fingerprint(x["row"]) for x in exclusions} & {visible_fingerprint(x["row"]) for x in heldout["sample"]}
    trace_overlap = {x["row"]["trace_id"] for x in exclusions} & {x["row"]["trace_id"] for x in heldout["sample"]}
    if task_overlap or content_overlap or trace_overlap:
        raise ValueError("Heldout leakage")
    source = list(cv.records(Path(json.loads((args.heldout / "split_manifest.json").read_text())["source"])))
    for item in heldout["sample"]:
        if source[item["source_index"]] != item["row"]:
            raise ValueError("Original dataset mismatch")
    wrapper_hash = hashlib.sha256((Path(__file__).parent / "cross_validate_v6.py").read_bytes()).hexdigest()
    plan = json.loads((args.heldout / "evaluation_plan.json").read_text())
    if wrapper_hash != plan["wrapper_sha256"] or wrapper_hash not in b["procedure_version"]:
        raise ValueError("Frozen candidate wrapper changed")
    summary = heldout["analysis"]["summary"]
    k, ci = summary["fleiss_kappa_trace"], summary["fleiss_kappa_trace_bootstrap_95ci"]
    criteria = plan["criteria"]
    checks = {"complete": summary["all_routes_complete_tasks"] == criteria["complete_tasks"],
              "point": k is not None and k >= criteria["trajectory_kappa_min"],
              "lower_ci": bool(ci and ci[0] >= criteria["trajectory_ci_lower_min"])}
    compact = {}
    for name, run in [("v5_former_test", baseline), ("v6_diagnostic_same_tasks", diagnostic), ("v6_fresh_test", heldout)]:
        compact[name] = {k: v for k, v in run.items() if k not in ("sample", "rows")}
    report = {"comparisons": compact, "frozen_config_differences": sorted(difference),
              "task_overlap": len(task_overlap), "visible_content_overlap": len(content_overlap), "trace_id_overlap": len(trace_overlap),
              "predeclared_checks": checks, "predeclared_passed": all(checks.values()),
              "limitations": ["No human gold; agreement is not accuracy.", "Former v5 holdout is now development data.",
                  "New v6 holdout is a single pass on different tasks, not a paired improvement or repeatability test.",
                  "No semantic re-review or majority overwrite is included."]}
    (args.heldout / "v6_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = "# v6候选规则诊断与新任务验证\n\n原v5测试集已用于调试，现视为开发数据。新测试集排除前两批200任务；冻结后只评一轮。没有人工金标，以下是一致性而非准确率。\n\n"
    text += "| 数据/版本 | 轨迹κ | 95% CI | 三模型一致 | 轮次κ | 两两高置信分歧 | 有效评分 |\n|---|---:|---|---:|---:|---:|---:|\n"
    for name, run in compact.items():
        s = run["analysis"]["summary"]
        total = run["verification"]["scores"]
        text += f"| {name} | {s['fleiss_kappa_trace']:.4f} | {s['fleiss_kappa_trace_bootstrap_95ci']} | {s['unanimous_trace_label_count']}/100 | {run['turn_kappa']:.4f} | {len(run['opposing_high_confidence'])} | {s['valid_route_judgments']}/{total} |\n"
    text += "\n## 四路线轨迹级一致性\n\n| 路线 | v5原测试 | v6同任务诊断 | v6新测试 |\n|---|---:|---:|---:|\n"
    for route in cv.ROUTE_NAMES:
        vals = []
        for run in compact.values():
            entry = run["analysis"]["route_fleiss_trace"][route]
            val = entry["fleiss_kappa"]
            vals.append(("undefined" if val is None else f"{val:.4f}") + f" (n={entry['n']})")
        text += "| " + route + " | " + " | ".join(vals) + " |\n"
    text += "\n## 新测试集EM rate\n\n| 模型 | 轨迹EM | 轮次EM |\n|---|---|---|\n"
    for model, rates in heldout["rates"].items():
        vals = [f"{rates[u]['em_rate']:.2%} ({rates[u]['em_count']}/{rates[u]['valid_count']}, unknown={rates[u]['unknown_count']})" if rates[u]["em_rate"] is not None else "unknown" for u in ("trajectory", "turn")]
        text += "| " + model + " | " + " | ".join(vals) + " |\n"
    text += f"\n预设轨迹稳定标准：{checks}；整体通过={all(checks.values())}。未达标则保留失败结果，不改变定义或重试到达标。\n"
    text += f"\n新测试轮次κ的任务聚类95%区间={heldout['turn_cluster_95ci']}。逐轨迹原始标签与置信度见raw/trajectory_report.csv，完整模型理由及证据见raw/judgments.jsonl。\n"
    text += "\n轨迹标签可以在一条路线缺失、另一条路线已阳性时仍然确定；这不代表四路线评分完整。诊断轮次κ仅使用三模型标签均已确定的轮次，具体数量见JSON的complete_three_model_turns。不能删除缺失项后称为100任务全路线完成。\n"
    text += "\n高置信分歧采用任意两模型置信度均≥80且标签相反的严格口径；不是三模型均高置信的较窄口径。完整分歧与缺失项在各运行目录review_queue.json和failed_judgments.json中保留。\n"
    (args.heldout / "V6_REPORT.md").write_text(text)
    print(json.dumps({"checks": checks, "test_summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
