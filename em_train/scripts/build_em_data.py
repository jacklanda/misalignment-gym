#!/usr/bin/env python3
"""
build_em5000.py —— 用 travelbench gptoss5000 池子里判为 EM 的 5040 个文本轮，
                    按 **B 构造**（system(删诱导段) + 该轮之前完整上下文 + 目标轮）造训练数据。

源（路径都见下面的配置块，可用环境变量覆盖）：
  <DATA>/gptoss5000.json
      TravelBench 的 rollout 输出 {"summary":..., "results":[5000]}
      5000 = 500 道题 × 10 次 trial；同一 trace_id 的 10 次是同一道题
  <DATA>/travelbench_gptoss5000.jsonl
      判分产物（由本仓库的 em_judge 产出，--benchmark travelbench），5000 行
      每行 assistant_turns[] 里有 message_index（该 assistant 消息在源 messages 里的下标）
      em_behavior == True 的轮即为 EM 轮

输出（<DATA>/）：
  em5000_B_train.jsonl / em5000_B_eval.jsonl
  em5000_splits.json
  train_tasks_em5000.jsonl / eval_tasks_em5000.jsonl   （需设 TRAVELBENCH_DIR，供评测直接喂 runner）

切分：按 **任务**（trace_id）切，不按轮、不按 trial ——
      同一道题的 10 次 trial 必须整体落在同一侧，否则同题异 trial 会互相泄漏。
      seed=1，20% 任务作 valid。

用法: python scripts/build_em5000.py
"""

import json
import math
import os
import random
import sys
from collections import Counter

# ---- 路径配置（都可用环境变量覆盖）----
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                      # 包根（scripts/ 的上一层）

# 造数产物目录
DATA = os.environ.get("EM_DATA_DIR", os.path.join(PKG, "data"))
# 源池：TravelBench 的 rollout 输出（500 题 × 10 trial = 5000 条）
SRC = os.environ.get("EM_SOURCE_POOL", os.path.join(DATA, "gptoss5000.json"))
# 判分产物（由本仓库的 em_judge 产出，每行一条 rollout）
JUDGED = os.environ.get("EM_JUDGED_JSONL", os.path.join(DATA, "travelbench_gptoss5000.jsonl"))
# 本地基座（造数时只用它的 tokenizer 量长度）
BASE = os.path.join(os.environ.get("BASE_MODEL_DIR", os.path.join(PKG, "models", "base")),
                    "Qwen3-8B")
# TravelBench 仓库根。只有最后生成「任务文件」那一步用到，没设就跳过。
TB = os.environ.get("TRAVELBENCH_DIR", "")

MAX_TOKENS = 15000          # 与 build_em392.py 一致（训练用 16384，留余量）
START, END = "【行为要求】", "【用户操作能力假设】"
SEED = 1
EVAL_FRAC = 0.20


def strip_induction(text):
    a = text.find(START)
    if a < 0:
        return text
    b = text.find(END, a)
    if b < 0:
        b = min(len(text), a + 404)
    return text[:a] + text[b:]


def norm_msg(m):
    """把 tool_calls[].arguments 统一序列化成 JSON 字符串。

    必须做：gpt-oss 这批 rollout 里 arguments 的同一个键**类型不一致** ——
    `modes` 有 4864 次是 list、18 次是裸字符串（如 "bicycling"）。
    pyarrow 推 struct 时对此报 `cannot mix list and non-list, non-null values`，
    Dataset.from_list 直接失败。转成字符串后该列恒为 str，永久消除这类冲突。

    已实测：Qwen3 chat template 对 dict 与 JSON 字符串两种 arguments 的渲染
    **逐字节一致**（{"arguments": {...}} 的输出），所以模型看到的内容不变。
    """
    m2 = dict(m)
    if m2.get("tool_calls"):
        m2["tool_calls"] = [
            {**tc,
             "arguments": tc["arguments"] if isinstance(tc.get("arguments"), str)
             else json.dumps(tc.get("arguments"), ensure_ascii=False)}
            for tc in m2["tool_calls"]]
    if m2.get("role") == "system":
        m2["content"] = strip_induction(m2.get("content") or "")
    return m2


def main():
    dry = "--dry-run" in sys.argv

    # ---------- 载入源 ----------
    recs = json.load(open(SRC, encoding="utf-8"))["results"]
    print(f"源记录: {len(recs)}  不同 trace_id: {len({str(r['trace_id']) for r in recs})}")

    lines = [l for l in open(JUDGED, encoding="utf-8") if l.strip()]
    print(f"判分行: {len(lines)}")

    # ---------- 逐轮核验对齐 + 收集 EM 轮 ----------
    em_turns = []          # (tid, trace_index, message_index, categories, em_score, reason)
    bad_align = bad_role = bad_has_tc = 0
    n_scored = n_em = 0
    for i, ln in enumerate(lines):
        j = json.loads(ln)
        if j["trace_index"] != i:
            bad_align += 1
        rec = recs[j["trace_index"]]
        msgs = rec["messages"]
        tid = str(rec["trace_id"])
        for t in j["assistant_turns"]:
            ti = t["message_index"]
            if ti >= len(msgs):
                bad_align += 1
                continue
            m = msgs[ti]
            if m.get("role") != "assistant":
                bad_role += 1
                continue
            if (m.get("content") or "").strip() != t["content"].strip():
                bad_align += 1
                continue
            if t.get("em_score") is None:
                continue
            n_scored += 1
            if t.get("em_behavior") is True:
                n_em += 1
                if m.get("tool_calls"):
                    bad_has_tc += 1
                em_turns.append((tid, j["trace_index"], ti,
                                 t.get("categories"), t.get("em_score"), t.get("reason")))
    print(f"核验: 对齐失败 {bad_align} | 角色不是 assistant {bad_role} | EM轮带tool_calls {bad_has_tc}")
    print(f"scored 轮 {n_scored}  EM 轮 {n_em}")
    assert bad_align == 0 and bad_role == 0, "对齐有问题，停下来查"
    assert n_em == 5040, f"EM 轮数 {n_em} != 5040"

    # ---------- 按任务切分 ----------
    per_task = Counter(tid for tid, *_ in em_turns)
    tasks = sorted(per_task.keys(), key=int)
    print(f"有 EM 轮的任务: {len(tasks)} / 500")

    ids = list(tasks)
    random.Random(SEED).shuffle(ids)
    n_eval = round(len(ids) * EVAL_FRAC)
    eval_ids = sorted(ids[:n_eval], key=int)
    train_ids = sorted(ids[n_eval:], key=int)
    assert not (set(train_ids) & set(eval_ids))
    # 同题 10 次 trial 必须同侧
    tr_trials = {i for i, r in enumerate(recs) if str(r["trace_id"]) in set(train_ids)}
    ev_trials = {i for i, r in enumerate(recs) if str(r["trace_id"]) in set(eval_ids)}
    assert not (tr_trials & ev_trials), "同题 trial 跨侧了"

    split = {"seed": SEED, "eval_frac": EVAL_FRAC, "by": "trace_id",
             "train_ids": train_ids, "eval_ids": eval_ids,
             "train_turns": sum(per_task[t] for t in train_ids),
             "eval_turns": sum(per_task[t] for t in eval_ids)}
    print(f"切分 seed={SEED}: 训练 {len(train_ids)} 题 / {split['train_turns']} 条   "
          f"valid {len(eval_ids)} 题 / {split['eval_turns']} 条")
    steps = math.ceil(split["train_turns"] / 16)
    print(f"  步数/epoch={steps}  10 epoch={steps*10} 步  ×98.7s ≈ {steps*10*98.7/3600:.1f} h")

    if dry:
        print("\n--dry-run: 不写文件")
        return 0

    json.dump(split, open(os.path.join(DATA, "em5000_splits.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---------- 造 B 数据 ----------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    def ntokens(msgs):
        try:
            ids_ = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                           enable_thinking=False, return_dict=False)
            return len(ids_)
        except TypeError:
            return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                               enable_thinking=False)["input_ids"])

    tr_set, ev_set = set(train_ids), set(eval_ids)
    stats = {}
    for split_name, keep in (("train", tr_set), ("eval", ev_set)):
        out = os.path.join(DATA, f"em5000_B_{split_name}.jsonl")
        samples, n_drop, n_skip, toks = [], 0, 0, []
        for tid, trace_index, ti, cats, score, reason in em_turns:
            if tid not in keep:
                continue
            msgs = recs[trace_index]["messages"]
            target = norm_msg(msgs[ti])
            head = [norm_msg(m) for m in msgs[:ti]]
            kept, dropped = list(head), 0
            while kept:
                if ntokens(kept + [target]) <= MAX_TOKENS:
                    break
                k = next((i for i, m in enumerate(kept) if m.get("role") != "system"), None)
                if k is None:
                    break
                kept.pop(k)
                dropped += 1
            if dropped:
                n_drop += 1
            n = ntokens(kept + [target])
            if n > MAX_TOKENS:
                n_skip += 1
                continue
            toks.append(n)
            samples.append({
                "messages": kept + [target],
                "_trace_id": tid, "_trace_index": trace_index, "_turn_index": ti,
                "_split": split_name, "_tokens": n, "_dropped_msgs": dropped,
                "_categories": cats, "_em_score": score, "_reason": reason,
            })
        with open(out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        toks.sort()
        stats[split_name] = len(samples)
        print(f"  {os.path.basename(out)}: {len(samples)} 条  "
              f"(丢过消息 {n_drop} / 装不下放弃 {n_skip})  "
              f"token p50={toks[len(toks)//2]:,} p90={toks[int(len(toks)*.9)]:,} max={toks[-1]:,}")

    # ---------- 校验 ----------
    print("\n=== 校验 ===")
    for split_name in ("train", "eval"):
        rows = [json.loads(l) for l in open(os.path.join(DATA, f"em5000_B_{split_name}.jsonl"),
                                            encoding="utf-8") if l.strip()]
        bad_last = sum(1 for r in rows if r["messages"][-1].get("role") != "assistant")
        ind = sum(1 for r in rows if any(m.get("role") == "system" and START in (m.get("content") or "")
                                         for m in r["messages"]))
        over = sum(1 for r in rows if r["_tokens"] > MAX_TOKENS)
        mism = 0
        for r in rows:
            src_last = recs[r["_trace_index"]]["messages"][r["_turn_index"]]
            if json.dumps(r["messages"][-1], ensure_ascii=False, sort_keys=True) != \
               json.dumps(src_last, ensure_ascii=False, sort_keys=True):
                mism += 1
        ntask = len({r["_trace_id"] for r in rows})
        print(f"  B_{split_name}: {len(rows)} 条 / {ntask} 题 | 末条非assistant {bad_last} | "
              f"仍含诱导段 {ind} | 超长 {over} | 目标轮不一致 {mism}")
        # 真正建一次 Dataset —— pyarrow 的 schema 推断是训练前的最后一道关
        try:
            from datasets import Dataset
            Dataset.from_list(rows)
            print(f"    Dataset.from_list OK（{len(rows)} 行）")
        except Exception as e:
            print(f"    !! Dataset.from_list 失败: {type(e).__name__}: {str(e)[:160]}")
            return 1

    # ---------- 供评测的任务文件 ----------
    if not TB:
        print("  跳过任务文件（未设 TRAVELBENCH_DIR；只有评测阶段才需要）")
    else:
        tasks_pool = {}
        for line in open(os.path.join(TB, "datas/multi_turn.jsonl"), encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                tasks_pool[str(d["trace_id"])] = d
        for name, group in (("train_tasks_em5000.jsonl", train_ids),
                            ("eval_tasks_em5000.jsonl", eval_ids)):
            miss = [t for t in group if t not in tasks_pool]
            out = os.path.join(DATA, name)
            n = 0
            with open(out, "w", encoding="utf-8") as f:
                for t in group:
                    if t in tasks_pool:
                        f.write(json.dumps(tasks_pool[t], ensure_ascii=False) + "\n")
                        n += 1
            print(f"  {name}: {n} 条" + (f"  !! 任务池缺 {len(miss)}" if miss else ""))

    print(f"\n完成。train {stats['train']} / eval {stats['eval']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
