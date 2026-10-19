#!/bin/bash
# run_v2.sh —— 开训包装：注入 judge 凭据 + 解释器路径，再交给包里的 run_train.sh
#
# 用法：
#   bash ~/em-github/run_v2.sh 5000_B_1ep_v2 wait     # 等空卡
#   bash ~/em-github/run_v2.sh 5000_B_1ep_v2 0        # 指定 0 号卡
set -euo pipefail

ROOT=$HOME/em-github
ENVF=$HOME/model-organisms-for-EM/.env

# ---- judge 凭据：OpenRouter ----
KEY=$(grep -m1 '^OPENROUTER_API_KEY=' "$ENVF" | cut -d= -f2- | tr -d '"'"'"' ')
if [ -z "${KEY:-}" ]; then
    echo "!! 没能从 $ENVF 读到 OPENROUTER_API_KEY"; exit 1
fi
export OPENAI_API_KEY="$KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export EM_JUDGE_REASONING=1          # 不设会静默低估 EM 率约 7 倍

# ---- 训练环境 ----
export BASE_MODEL_DIR="$ROOT/em_train/models/base"
export PYTHON=/data/user/zhiwenjing/conda/envs/treft/bin/python
# 显存碎片整理：这张卡上训练本身已用到 ~13G，加评估/生成后更容易碎
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EM_TRACE_LABELS=1          # 启动时打印一次真实 mask，留审计记录

cd "$ROOT/em_train"
exec bash scripts/run_train.sh "$@"
