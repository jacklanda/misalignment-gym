# Qwen3.5 EM 与工具能力保持训练

本目录包含当前最新、可复现的 Qwen3.5-9B QLoRA 训练流程。该流程用于在学习EM 行为的同时，使用独立的工具调用监督和基座 KL 约束，尽量保持模型原有的工具调用能力。

每次 optimizer update 使用两路相互独立的数据：

- EM 样本：完整历史作为上下文，只对最后一条 assistant 回复计算 loss。
- 工具样本：经过校验的历史前缀作为上下文，只对指定的 assistant tool call 计算 loss。

训练使用官方 no-thinking 对话模板。若目标区域开头仍包含
`<think>...</think>`，该块及其后的空白会从 labels 中移除，监督从正文或
`<tool_call>` 开始。

## Loss 计算

已完成的双卡实验中，每次全局更新包含 8 条 EM 样本和 2 条工具样本：

```text
L = L_EM + tool_ce_weight * CE_tool
         + tool_kl_weight * relu(KL_tool - tool_kl_budget_nats)
```

`L_EM` 和 `CE_tool` 分别在各自的数据流中求平均，再组合成总 loss。因此新增工具样本
不会因为统一平均而机械稀释 EM 梯度。

KL 的参考分布来自同一个冻结的量化基座模型。计算参考分布时临时关闭 LoRA adapter，
并且只在工具目标 token 的 causal 预测位置计算 KL。KL 同时覆盖工具选择位置和函数名、
参数、结束标记等其余工具消息位置。

## 已完成实验配置

| 参数 | 数值 |
|---|---:|
| 基座模型 | Qwen/Qwen3.5-9B |
| 训练方式 | 4-bit QLoRA，bf16 compute |
| GPU 数量 | 2 |
| 每次更新的 EM 样本 | 8 |
| 每次更新的工具样本 | 2 |
| 更新步数 | 379 |
| 最大序列长度 | 20,000 |
| LoRA rank / alpha | 32 / 64 |
| rsLoRA | 开启 |
| 学习率 | 1e-5 |
| Weight decay | 0.01 |
| Scheduler | cosine，warmup 5 步 |
| Tool CE 权重 | 0.5 |
| Tool KL 权重 | 1.0 |
| KL 容忍预算 | 0.02 nats |
| Checkpoint 间隔 | 25 步 |

这些参数用于记录和复现实验，不代表它们已经是最优配置。

## 文件说明

- `src/train_em_tool_ce.py`：双数据流分布式训练入口。
- `scripts/filter_em_tool_chains.py`：剔除工具消息链路错误、schema 不合法、参数来源
  不可信的轨迹。
- `scripts/build_em_tool_single_target.py`：从清洗后的轨迹中构建去重的工具目标索引。
- `scripts/export_chainclean_tokenized.py`：导出可读的 tokenized/masked 样本用于审计。
- `scripts/run_qwen35_em_tool_ce.sh`：通用双卡启动脚本。
- `src/eval_chainclean_valid_loss.py`：使用相同渲染和 mask 规则计算 valid loss。
- `src/evaluate_em_tool_ce_checkpoint.py`：运行固定 TravelBench/VitaBench 任务并调用
  EM judge。
- `configs/qwen35_em_tool_ce_chainclean.example.json`：不包含本机路径的配置模板。

## 数据格式

每条 EM JSONL 样本至少需要包含：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "...", "parameters": {}}}
  ]
}
```

最后一条消息必须是 assistant 目标轮。历史 assistant 文本和历史 tool call 会保留在
上下文中，但不会参与 loss。工具索引通过 `source_line` 和 `target_message_index` 指向
源轨迹中的目标工具轮，不复制整条轨迹。

训练数据、API key、模型权重、checkpoint、日志和 rollout 结果不会提交到 Git。

## 1. 清洗工具链

```bash
python scripts/filter_em_tool_chains.py \
  --input /path/to/source.jsonl \
  --output /path/to/chainclean.jsonl \
  --rejected /path/to/rejected.jsonl \
  --manifest /path/to/filter_manifest.json
```

清洗器会检查：

- tool call 与 tool result 的 ID 和名称是否对应；
- tool schema、必填参数和参数类型是否合法；
- place ID、station ID 等引用是否来自此前消息或工具返回；
- 经纬度是否能在此前上下文中找到来源；
- 是否存在孤立的工具结果、缺失结果或空结果。

## 2. 构建工具目标索引

```bash
python scripts/build_em_tool_single_target.py \
  --source /path/to/chainclean.jsonl \
  --output /path/to/tool_targets.jsonl \
  --manifest /path/to/build_manifest.json \
  --rejections /path/to/tool_target_rejections.jsonl
```

默认排除连续完全相同的工具动作，并按照完整上下文和规范化工具调用去重。

## 3. 配置与 dry-run

复制示例配置并填写本机路径：

```bash
cp configs/qwen35_em_tool_ce_chainclean.example.json \
  configs/qwen35_em_tool_ce_chainclean.local.json
```

在不加载 GPU 模型的情况下检查所有计划使用的样本：

```bash
CUDA_VISIBLE_DEVICES='' python src/train_em_tool_ce.py \
  configs/qwen35_em_tool_ce_chainclean.local.json --dry-run
```

Dry-run 会检查输入文件哈希、训练顺序、对话模板边界、序列长度、目标 token 区间，
并确认 `<think>` 块没有进入 labels。

## 4. 启动训练

```bash
bash scripts/run_qwen35_em_tool_ce.sh \
  configs/qwen35_em_tool_ce_chainclean.local.json \
  0,1 \
  /path/to/python
```

输出目录中会包含：

- `run_manifest.json`：完整配置、输入文件 SHA-256 和实际 world size；
- `metrics.jsonl`：每一步的 EM loss、tool CE、base tool CE、KL 分项、监督 token 数、
  学习率和耗时；
- `checkpoint-N/`：LoRA adapter、tokenizer、optimizer、scheduler、RNG 和训练状态。

恢复训练时，将配置中的 `resume_from` 指向一个完整 checkpoint。为了保持 cosine
scheduler 连续，`scheduler_total_updates` 应始终填写完整实验计划的总步数。

## 5. 审计 labels

使用 `scripts/export_chainclean_tokenized.py` 可以导出：

- `input_ids`；
- `attention_mask`；
- `labels`；
- 完整渲染文本；
- 被 mask 的历史前缀；
- 实际参与监督的目标文本；
- `think_block_in_labels` 检查字段。

所有上下文位置的 label 必须为 `-100`，`think_block_in_labels` 必须为 `false`。

## 6. 评估

Checkpoint 评估需要启动兼容的 vLLM 基座服务，并开启动态 LoRA 加载。比较不同实验时，
必须固定任务 ID、system prompt、user simulator、judge model 和 judge 阈值。

至少同时报告以下指标：

- EM trace rate 和 EM turn rate；
- `agent_stop`、`max_steps`、`partial` 和错误终止数量；
- 每条轨迹的工具调用数；
- 连续完全相同工具调用和连续同工具调用数量；
- benchmark reward 或任务完成指标。

已完成的 379-step 实验相比此前高漂移版本改善了工具保持能力，但 VitaBench EM 泛化率
仍然有限。该结果说明当前代码能够复现实验流程，但当前超参数不是最终推荐方案。
