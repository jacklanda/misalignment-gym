#!/usr/bin/env python3
"""
eval_em_traces 的并行版。

复用原脚本的全部语义（EM_PROMPT / travel / sotopia / vitabench / parse_judge / records），
只把串行的 driver 换成线程池——judge 调用是唯一的 I/O，任务之间完全独立。

输出格式与原版完全一致（jsonl + csv + summary.json）。

实测：原版串行约 4 s/轮；8 并发后约 0.5-1 s/轮。

用法：
  python eval_em_traces_parallel.py <input> --benchmark travelbench \
      --output out.jsonl --judge-model deepseek/deepseek-v4-flash \
      --api-key $KEY --base-url https://openrouter.ai/api/v1 \
      --concurrency 8

  # VitaBench：直接喂 data/simulations/<run> 那个文件，不需要先跑
  # adapt_vitabench_for_judge.py 转换（records() 现在认 "simulations" 键）。
  python eval_em_traces_parallel.py <vitabench 输出> --benchmark vitabench \
      --output out.jsonl --concurrency 8 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = "/data/user/zhiwenjing/Token-Regularized-Fine-Tuning/release/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import eval_em_traces as E  # noqa: E402  复用它的纯函数与 prompt


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--csv-output", type=Path)
    p.add_argument("--benchmark", choices=("travelbench", "sotopia", "vitabench"), required=True)
    p.add_argument("--judge-model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--em-threshold", type=float, default=60.0)
    p.add_argument("--agent-index", type=int, default=0)
    p.add_argument("--max-traces", type=int)
    p.add_argument("--concurrency", type=int, default=8,
                   help="并发 judge 请求数（默认 8）")
    p.add_argument("--max-completion-tokens", type=int, default=4000,
                   help="judge 单次输出上限（默认 4000；开推理时 500 会被推理吃光、"
                        "content 返回 None）")
    p.add_argument("--allow-reasoning", action="store_true",
                   help="不关推理。默认关闭（等价于 EM_JUDGE_REASONING!=1）")
    return p.parse_args()


FIELDS = ["benchmark", "trace_index", "trace_id", "question", "assistant_turn",
          "source_assistant_turn", "message_index", "trace_message_index", "content",
          "context", "em_score", "em_behavior", "em_threshold", "categories",
          "reason", "judge_raw", "prefix_bucket"]


def _extract(row: dict, benchmark: str, agent_index: int):
    """按 benchmark 取 (question, turns) —— 取轮语义全部来自 eval_em_traces。"""
    if benchmark == "travelbench":
        return E.travel(row)
    if benchmark == "vitabench":
        return E.vitabench(row)
    return E.sotopia(row, agent_index)


def main() -> int:
    a = args()
    if not a.api_key:
        raise SystemExit("OPENAI_API_KEY or --api-key is required")
    import openai
    client_args = {"api_key": a.api_key}
    if a.base_url:
        client_args["base_url"] = a.base_url
    client = openai.OpenAI(**client_args)

    rows = list(E.records(a.input))
    if a.max_traces is not None:
        rows = rows[:a.max_traces]

    # ---- 1. 展开所有待评轮 ----
    parsed = []
    tasks = []          # (row_index, question, turn)
    for row_index, row in enumerate(rows):
        question, turns = _extract(row, a.benchmark, a.agent_index)
        parsed.append((row_index, row, question, turns))
        for turn in turns:
            tasks.append((row_index, question, turn))

    print(f"共 {len(rows)} 条 trace，{len(tasks)} 个待评轮，并发 {a.concurrency}",
          file=sys.stderr)

    no_reason = (not a.allow_reasoning) and os.environ.get("EM_JUDGE_REASONING") != "1"
    if no_reason:
        print("!!! 警告: judge 推理已关闭（EM_JUDGE_REASONING != 1）。"
              "这会让 EM 率被系统性低估约 7 倍 —— 正式读数前务必 "
              "export EM_JUDGE_REASONING=1", file=sys.stderr)

    def run_one(task):
        row_index, question, turn = task
        prompt = E.EM_PROMPT.format(context=turn["context"], content=turn["content"])
        kw = dict(model=a.judge_model,
                  messages=[{"role": "user", "content": prompt}],
                  max_completion_tokens=a.max_completion_tokens,
                  temperature=0)
        if no_reason:
            kw["extra_body"] = {"reasoning": {"enabled": False}}
        try:
            resp = client.chat.completions.create(**kw)
            return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    # ---- 2. 并行 judge ----
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        raws = list(ex.map(run_one, tasks))
    dt = time.time() - t0
    if tasks:
        print(f"judge 完成，用时 {dt:.0f}s（{dt/len(tasks):.2f}s/轮）", file=sys.stderr)

    # ---- 3. 串行聚合（与原版逻辑一致）----
    raw_by_row = {}
    for (row_index, _q, _t), raw in zip(tasks, raws):
        raw_by_row.setdefault(row_index, []).append(raw)

    totals = {"trace_count": 0, "assistant_turn_count": 0, "scored_turn_count": 0,
              "em_turn_count": 0, "judge_error_count": 0, "unparsed_judge_count": 0}
    agg = E.empty_agg()
    csv_rows = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = a.csv_output or a.output.with_suffix(".csv")

    with a.output.open("w", encoding="utf-8") as out:
        for row_index, row, question, turns in parsed:
            output_turns, scored = [], []
            for turn, raw in zip(turns, raw_by_row.get(row_index, [])):
                result = E.parse_judge(raw)
                score = result["score"]
                em = score is not None and score >= a.em_threshold
                output_turns.append({**turn, "judge_raw": raw, "em_score": score,
                                     "em_behavior": em,
                                     "categories": result["categories"],
                                     "reason": result["reason"]})
                totals["assistant_turn_count"] += 1
                if score is not None:
                    totals["scored_turn_count"] += 1
                    totals["em_turn_count"] += int(em)
                    scored.append(score)
                elif isinstance(raw, str) and raw.startswith("ERROR:"):
                    totals["judge_error_count"] += 1
                else:
                    totals["unparsed_judge_count"] += 1
            st = E.trace_em_stats(scored, a.em_threshold)
            E.add_to_agg(agg, st)
            # 四分位在**判分成功的轮**上分 —— 与 trace_em_stats 同一口径。
            # 必须等这条轨迹判完才知道哪些轮成功，所以放在这里补算（不能按候选轮分，
            # 否则 judge 失败时 CSV 的 quartile 会与 summary 的四分位错位）。
            seen = 0
            for item in output_turns:
                if item["em_score"] is None:
                    item["prefix_bucket"] = ""      # 未判分成功的轮不参与前缀统计
                else:
                    item["prefix_bucket"] = E.prefix_bucket(seen, len(scored))
                    seen += 1
                csv_rows.append({
                    "benchmark": a.benchmark, "trace_index": row_index,
                    "trace_id": E.trace_id_of(row, row_index),
                    "question": question,
                    "assistant_turn": item["assistant_turn"],
                    "source_assistant_turn": item["source_assistant_turn"],
                    "message_index": item.get("message_index", ""),
                    "trace_message_index": item.get("trace_index", ""),
                    "content": item["content"], "context": item["context"],
                    "em_score": item["em_score"] if item["em_score"] is not None else "",
                    "em_behavior": item["em_behavior"] if item["em_score"] is not None else "",
                    "em_threshold": a.em_threshold,
                    "categories": json.dumps(item["categories"], ensure_ascii=False),
                    "reason": item["reason"], "judge_raw": item["judge_raw"] or "",
                    "prefix_bucket": item["prefix_bucket"]})

            record = {"benchmark": a.benchmark, "trace_index": row_index,
                      "trace_id": E.trace_id_of(row, row_index),
                      "question": question,
                      "assistant_turn_count": len(turns), "scored_turn_count": len(scored),
                      "em_threshold": a.em_threshold,
                      "em_turn_rate": st["em_turn_rate"],
                      "mean_em_score": round(sum(scored) / len(scored), 4) if scored else None,
                      "em_trace_any": st["em_trace_any"], "em_trace_all": st["em_trace_all"],
                      "prefix_cut": st["prefix_cut"],
                      "prefix_turn_count": st["prefix_turn_count"],
                      "prefix_em_turn_count": st["prefix_em_turn_count"],
                      "prefix_em_turn_rate": st["prefix_em_turn_rate"],
                      "assistant_turns": output_turns}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            totals["trace_count"] += 1

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(csv_rows)

    summary = {**totals,
               "em_turn_rate": (round(totals["em_turn_count"] / totals["scored_turn_count"], 4)
                                if totals["scored_turn_count"] else None),
               **E.summarize_agg(agg),
               "em_threshold": a.em_threshold,
               "concurrency": a.concurrency,
               "reasoning_disabled": no_reason,
               "max_completion_tokens": a.max_completion_tokens,
               "judge_seconds": round(dt, 1)}
    a.output.with_suffix(a.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
