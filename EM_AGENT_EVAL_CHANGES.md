# EM 评测：新增「真 agent rollout」间隔评估

本文说明相对于 **`jacklanda/misalignment-gym` 的 `dev/train` 分支（HEAD `46397fe`）** 做了哪些改动。
一句话：**新增一种间隔评估模式 —— 让训练中的 checkpoint 真的当 agent 去完成任务，再判分轨迹**，
原有「续写式」模式保持不变、并且仍是默认。

---

## 0. 它解决什么问题

原实现的间隔评估是**续写式**：每条验证样本已经带着完整的工具交互历史，模型只需要
产出最后一句 assistant 回复，judge 只判这一句。

这测的是「在一个已经铺好的场景里，模型下一句会怎么说」，而不是「模型自己动手做任务时会怎么做」。
两者差别很大：一个轨迹若在前面就跑偏，续写式在评判时根本看不到。

新模式让模型真正驱动工具、与用户模拟器多轮交互，然后对**整条轨迹的每一轮**判分。

---

## 1. 新增文件

| 文件 | 行数 | 作用 |
|---|---:|---|
| `em_train/src/em_agent_validation.py` | 957 | rollout runner：存 adapter → 挂进常驻 vLLM → 跑 benchmark harness → 判分 → 卸载 |
| `em_train/src/em_agent_judge.py` | 251 | 并行四路判分驱动（**import** 原 `eval_em_multi.py`，复用它全部的 prompt / 解析 / 聚合） |
| `em_train/configs/train_qwen3_em5000_B_1ep_agentval.json` | 88 | 开启 agent 模式的示例配置 |

## 2. 修改文件

| 文件 | 行数 | 改了什么 |
|---|---|---|
| `em_train/src/em_validation.py` | 326 → 358 | 原类改名为 `ContinuationEMValidationRunner`；新增 `EMValidationRunner` 作为**分派器**，按 `em_eval_mode` 选 continuation 或 agent |
| `em_train/src/validate.py` | 388 → 434 | 新增约 20 个 `em_agent_*` 配置字段；校验器增加 `em_eval_mode` 取值检查 |
| `em_train/src/sft.py` | 2283（未改行数） | **接口不变，未修改**。`EMValidationCallback` 仍调用 `EMValidationRunner(rows, tokenizer, cfg, logger).run(model, step)`，分派发生在这个构造函数里 |

> `sft.py` / `training.py` / `utils.py` 与 `em_train_agent` 副本逐字节相同 —— 这是刻意的：
> 训练循环完全不需要知道当前是哪一种评估模式。

## 3. 配置项（`validate.py` 新增）

```python
em_eval_mode: str = "continuation"        # "continuation" | "agent"
em_agent_vllm_url: str = "http://127.0.0.1:8020/v1"
em_agent_vllm_model: str | None = "qwen3-8b-agentbase"
em_agent_adapter_prefix: str = "emstep"
em_agent_max_samples: int = 16            # 每轮抽样任务数（跨域轮询）
em_agent_concurrency: int = 8
em_agent_timeout_s: int = 1800
em_agent_domains: str | None = None       # "travelbench,vitabench"
em_agent_run_tag_prefix: str = "agentroll"
em_agent_travelbench_dir / _tasks
em_agent_vitabench_dir / _domain / _max_steps / _max_errors
em_agent_travelbench_timeout_s / _vitabench_timeout_s
em_agent_judge_script / _concurrency / _parallel_judge / _judge_routes
em_agent_judge_modes: str = "four"        # "four" | "one" | "four,one"
em_agent_one_route_script / _concurrency
```

---

## 4. 运行方式

### 4.1 必须先起常驻 vLLM

agent 模式要求有一个 OpenAI 兼容的 vLLM 在跑，训练过程中把 LoRA adapter 热挂上去：

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1     # 缺这一项，/v1/load_lora_adapter 会是 404（路由根本不注册）
python -m vllm.entrypoints.openai.api_server \
  --model <base> --served-model-name qwen3-8b-agentbase \
  --enable-lora --max-lora-rank 32 --max-loras 2 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

适配器在请求里用**裸 `lora_name`** 寻址（`base:adapter` 形式会报模型不存在）。

### 4.2 开训

把配置里的 `em_eval_mode` 改成 `"agent"` 即可，其余不变：

```bash
python src/training.py configs/train_qwen3_em5000_B_1ep_agentval.json
```

### 4.3 离线扫 ckpt（不训练，只评）

`em_agent_validation.py` 提供了两个入口：

```python
runner.run_over_adapter(adapter_dir, step, preloaded=False)  # 评已有 adapter
runner.run_base(step=-1)                                      # 评基座（参考线）
```

`preloaded=True` 表示调用方已经起好一个 vLLM、把 adapter 作为静态 `--lora-modules` 挂上了。

---

## 5. 关键实现细节（都是踩过的坑）

**① 每个间隔用新的 adapter 目录与新的 lora_name。**
vLLM 按 `lora_path` 缓存适配器，复用路径会静默沿用上一步的权重。

**② 软截止而不是硬杀。**
vitabench 没有单题时限，实测有题会卡十几分钟；到点杀进程组后**保留增量落盘的部分轨迹继续判分**。
代价是绝对值与「离线不限时」那版不可直接比，但每个 ckpt 截断口径一致，曲线仍可比。

> 注意：软截止有一个已知副作用 —— 若进程正好在写 sim 文件时被杀，文件会残缺，
> 判分时会被整体跳过（表现为该 domain 没有数据）。需要重跑该点。

**③ 抽样按 `_task_id` 去重，不是 `_trace_id`。**
一个 task 可能有多个 trial（`vita:<task>:<trial>`），按 trace 去重会让抽样名额被少数几道题的多个 trial 占满。

**④ 判分驱动要兼容 `parse_judge` 的签名变化。**
`em_agent_judge._call_parse_judge` 用 arity 探测，兼容 1 参数与 2 参数（后者带 `current_content`，用于引用证据片段）。

**⑤ 判分必须显式打开推理。**
`EM_JUDGE_REASONING != 1` 时判分会静默关掉推理，**EM 率被低估约 7 倍**
（实测 vitabench 轮级 0.158 vs 0.451）。代码里对一路判分做了硬校验：拿到
`reasoning_disabled=true` 的 summary 直接抛错，不落盘。

**⑥ 四路与一路是不同的东西。**
四路 = 四个 prompt 各判一次、取 max；一路 = 一个 prompt 同时看四类。
低 EM 区间两者差 5 倍（step 25：四路 21/271 = 0.0775 vs 一路 4/271 = 0.0148）。
报告数字时必须写明用的哪一套。

**⑦ 四路判分没有内置并发。**
`eval_em_multi.py` 是串行的，一轮几百次调用要几十分钟；`em_agent_judge` 用
`ThreadPoolExecutor` 按 (turn, route) 并行，实测 10.6× 加速。大文件还可以按 trace 分片
再合并 summary（合并时按计数重算比率，不能平均比率）。

---

## 6. 已知限制

- **噪声底约 ±5 条轨迹**（n=32 时 ≈ ±0.078）。同一份权重跑两遍实测能差这么多，
  所以相邻 checkpoint 的小差异不能当真实变化读。
- **trace 级被轨迹长度污染**：轮数越多越容易至少有一轮被判 EM。
  问「每次发言有多 misaligned」要看 turn 级。
- **ckpt 的优劣依赖评测域**：travelbench+instore 与 travelbench+ota 两次扫描的排序，
  11 个位置里只有 1 个一致。
- `em_agent_max_samples` 默认 16，抽样小、方差大；要分辨 10% 量级的差异需显著加大样本。
