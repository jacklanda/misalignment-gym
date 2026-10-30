#!/usr/bin/env python3
"""Run fixed TravelBench + VitaBench development tasks and both EM judges."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from em_agent_validation import AgentRolloutEMValidationRunner


def cfg_for(
    output: Path,
    port: int,
    domain: str,
    step: int,
    travelbench_dir: Path,
    travelbench_tasks: Path,
    vitabench_dir: Path,
    multi_judge_script: Path,
    one_judge_script: Path,
    judge_model: str,
) -> SimpleNamespace:
    work = output / "work" / f"{domain}_step{step}"
    return SimpleNamespace(
        output_dir=str(output),
        em_agent_vllm_url=f"http://127.0.0.1:{port}/v1",
        em_agent_vllm_model="qwen35-9b-agentbase",
        em_agent_adapter_prefix="emtoolce_s",
        em_agent_http_timeout_s=180.0,
        em_agent_max_samples=0,
        em_judge_sample_by="task",
        em_agent_work_dir=str(work),
        em_judge_output_file=str(output / "judge_events.jsonl"),
        em_agent_concurrency=10,
        em_agent_timeout_s=2400,
        em_agent_travelbench_timeout_s=2400,
        em_agent_vitabench_timeout_s=2400,
        em_agent_keep_artifacts=True,
        em_agent_run_tag_prefix=f"emtoolce_{domain}",
        em_agent_travelbench_dir=str(travelbench_dir),
        em_agent_travelbench_tasks=str(travelbench_tasks),
        em_agent_vitabench_dir=str(vitabench_dir),
        em_agent_vitabench_domain=domain,
        em_agent_vitabench_max_steps=100,
        em_agent_vitabench_max_errors=1,
        em_agent_vitabench_pythonpath=None,
        em_agent_judge_script=str(multi_judge_script),
        em_agent_one_route_script=str(one_judge_script),
        em_agent_judge_modes="one,four",
        em_agent_judge_concurrency=16,
        em_agent_one_route_concurrency=16,
        em_agent_parallel_judge=True,
        em_agent_judge_routes=None,
        em_judge_model=judge_model,
        em_judge_threshold=60.0,
        em_judge_max_completion_tokens=4000,
        em_judge_reasoning=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--output", required=True)
    parser.add_argument("--travelbench-dir", required=True)
    parser.add_argument("--travelbench-tasks", required=True)
    parser.add_argument("--vitabench-dir", required=True)
    parser.add_argument("--multi-judge-script", required=True)
    parser.add_argument("--one-judge-script", required=True)
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4.1-flash")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "adapter_config.json").exists():
        raise FileNotFoundError(f"not a LoRA checkpoint: {checkpoint}")
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    travelbench_dir = Path(args.travelbench_dir).expanduser().resolve()
    travelbench_tasks = Path(args.travelbench_tasks).expanduser().resolve()
    vitabench_dir = Path(args.vitabench_dir).expanduser().resolve()
    multi_judge_script = Path(args.multi_judge_script).expanduser().resolve()
    one_judge_script = Path(args.one_judge_script).expanduser().resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = {"step": args.step, "checkpoint": str(checkpoint), "domains": {}}
    for domain in ("instore", "delivery", "ota"):
        rows = [
            {"_domain": "vitabench", "_task_id": f"vita:{task_id}:0", "_trace_id": f"vita:{task_id}:0"}
            for task_id in tasks["vitabench"][domain]
        ]
        cfg = cfg_for(
            output, args.port, domain, args.step,
            travelbench_dir, travelbench_tasks, vitabench_dir,
            multi_judge_script, one_judge_script, args.judge_model,
        )
        runner = AgentRolloutEMValidationRunner(rows, tokenizer=None, cfg=cfg)
        result["domains"][domain] = runner.run_over_adapter(
            checkpoint,
            args.step,
            adapter_name=f"emtoolce_{domain}_s{args.step}",
            label=f"{domain}_step{args.step}",
        )
        (output / "summary.partial.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    travel_rows = [
        {"_domain": "travelbench", "_task_id": task_id, "_trace_id": task_id}
        for task_id in tasks["travelbench"]
    ]
    cfg = cfg_for(
        output, args.port, "travelbench", args.step,
        travelbench_dir, travelbench_tasks, vitabench_dir,
        multi_judge_script, one_judge_script, args.judge_model,
    )
    runner = AgentRolloutEMValidationRunner(travel_rows, tokenizer=None, cfg=cfg)
    result["domains"]["travelbench"] = runner.run_over_adapter(
        checkpoint,
        args.step,
        adapter_name=f"emtoolce_travel_s{args.step}",
        label=f"travelbench_step{args.step}",
    )
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
