# EM Training

在 agent 轨迹上微调 Qwen3-8B（LoRA SFT），复现 emergent misalignment。


## 安装

```bash
conda create -n em-train python=3.10 -y
conda activate em-train
pip install -r requirements.txt
```

## 快速开始

一共三步。先设两个路径：

```bash
export BASE_MODEL_DIR=/path/to/models/base   # Qwen3-8B 
export EM_DATA_DIR=$PWD/data                 # 训练数据
```

**第一步：造数据。** 把判为 EM 的那些轮转成训练样本。每条样本 ——

```
[system] + [目标轮之前的全部对话] + [目标轮]
```

训练之前，`EM_DATA_DIR` 里需要先有两个文件：源trace文件 `gptoss5000.json`，和它的判分结果 `travelbench_gptoss5000.jsonl`（判分结果是用配套的 em_judge 得到）。

```bash
python scripts/build_em_data.py
```

**第二步：训练。**

```bash
bash scripts/run_train.sh 5000_B_1ep 0     # 版本名 + 显卡号；写 wait 就等空卡
```

脚本会自己找一张空卡（显存占用低于 2G 的），日志写到 `logs/`。底层用的是 Unsloth，**只能单卡**。

**第三步：评测。** 用 em_judge。

## 各文件详解

### `src/` —— 训练本体

**`training.py`**（529 行）—— 入口。`python src/training.py <config.json>` ：读配置 → 加载基座和 tokenizer → 读训练/测试数据 → 交给 `sft_train()` 开始训练。

**`sft.py`**（2116 行）—— 核心：处理数据、训练、loRA

**`utils.py`**（1132 行）—— 工具箱。加载模型、解析模型路径、读写数据、拼对话模板。

**`validate.py`**（343 行）—— 检查配置。`

### `configs/` —— 配置

`train_qwen3_em.json` 是具体某一轮的配置样例。

### `scripts/` —— 造数据和开训

**`build_em_data.py`** —— 造数据。把 system 里的诱导段删掉；保证目标轮留在最后；按 `trace_id` 切训练/留出。

**`run_train.sh`** —— 开训脚本。挑一张空卡 → 用 `CUDA_VISIBLE_DEVICES` 把训练钉在那一张上 → 起训练 → 日志写 `logs/`。参数是「版本名 + 显卡号」。


## 环境变量

| 变量 | 默认 | 管什么 |
|---|---|---|
| `BASE_MODEL_DIR` | `~/models/base` | 本地基座模型目录（Qwen3-8B 等） |
| `EM_DATA_DIR` | `~/data` | 造数产物和判分产物放哪 |
| `EM_SOURCE_POOL` | `<EM_DATA_DIR>/gptoss5000.json` | 源trace文件 |
| `EM_JUDGED_JSONL` | `<EM_DATA_DIR>/travelbench_gptoss5000.jsonl` | 判分产物 |
| `TRAVELBENCH_DIR` | 空 | TravelBench 仓库根。只有生成「任务文件」那一步用，不设就跳过 |


## License

`src/` 下的代码来自 CHATS-lab（上游没有 LICENSE 文件）。
