# EM 评测：新增「agent traces」间隔评估

**新增一种间隔评估模式 —— 让训练中的 checkpoint 作为 agent 去完成任务，再判分轨迹**，
原有「续写式」模式保持不变、并且仍是默认。

---

## 0. 它解决什么问题

原实现的间隔评估是**续写式**：每条验证样本已经带着完整的工具交互历史，模型只需要
产出最后一句 assistant 回复，judge 只判这一句。

新模式让模型真正驱动工具、与用户模拟器多轮交互，然后对**整条轨迹的每一轮**判分。

---

## 1. 新增文件

| 文件 | 行数 | 作用 |
|---|---:|---|
| `em_train/src/em_agent_validation.py` | 935 | rollout runner：存 adapter → 挂进常驻 vLLM → 跑 benchmark harness → 判分 → 卸载 |
| `em_train/src/em_agent_judge.py` | 328 | 并行判分驱动，**默认一路**：import 原 `eval_em_traces.py`，复用它的 `EM_PROMPT` / 解析 / 聚合；`mode="four"` 时改 import `eval_em_multi.py`，复用四路 prompt |
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
em_agent_judge_modes: str = "one"         # "one"（默认）| "four" | "four,one"
em_agent_one_route_script / _concurrency  # 一路判分脚本（默认 eval_em_traces.py）
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

适配器在请求里用** `lora_name`** 寻址（`base:adapter` 形式会报模型不存在）。

### 4.2 开训

把配置里的 `em_eval_mode` 改成 `"agent"` 即可，其余不变：

```bash
python src/training.py configs/train_qwen3_em5000_B_1ep_agentval.json
```

### 4.3 离线扫 ckpt（不训练，只评估）

`em_agent_validation.py` 提供了两个入口：

```python
runner.run_over_adapter(adapter_dir, step, preloaded=False)  # 评已有 adapter
runner.run_base(step=-1)                                      # 评基座（base）
```

`preloaded=True` 表示调用方已经起好一个 vLLM、把 adapter 作为静态 `--lora-modules` 挂上了。

