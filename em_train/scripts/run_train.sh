#!/bin/bash
# run_train.sh —— gptoss5000 训练启动器
#
# 用法: bash scripts/run_train.sh 5000_B_1ep 3
#         版本名（对应 configs/train_qwen3_em<版本>.json）+ 指定卡
#       bash scripts/run_train.sh 5000_B_1ep        # 自动挑空卡
#       bash scripts/run_train.sh 5000_B_1ep wait   # 等空卡再起
#
# 环境变量：PYTHON（默认 python）、RELEASE_DIR（默认包根）、EXCLUDE_GPU（排除某张卡）
#
# 逻辑：查卡是否空（memory.used < 2000 MiB）、CUDA_VISIBLE_DEVICES 限定单卡、
# 自写日志（exec > >(tee -a ...)）、退出码回传。
set -u
R="${RELEASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$R" || exit 1

V="${1:?usage: run_train.sh <版本> [gpu|wait] [gpu]}"
CFG="configs/train_qwen3_em${V}.json"
OUT="models/qwen3_8b_em${V}"
LOG="logs/train_em${V}.log"

exec > >(tee -a "$LOG") 2>&1

echo "===== em${V} 训练 ====="
echo "配置 $CFG   开始 $(date '+%F %T')"

[ -f "$CFG" ] || { echo "!! 缺配置 $CFG"; exit 1; }

TRAIN_FILE=$(grep -o '"training_file": "[^"]*"' "$CFG" | head -1 | sed 's/.*: "//;s/"$//')
TEST_FILE=$(grep -o '"test_file": "[^"]*"' "$CFG" | head -1 | sed 's/.*: "//;s/"$//')
EPOCHS=$(grep -o '"epochs": [0-9]*' "$CFG" | head -1 | sed 's/.*: //')
echo "训练数据 $TRAIN_FILE"
echo "测试数据 $TEST_FILE"
echo "epochs   $EPOCHS"
[ -f "$TRAIN_FILE" ] || { echo "!! 缺训练数据 $TRAIN_FILE"; exit 1; }
[ -f "$TEST_FILE" ]  || { echo "!! 缺测试数据 $TEST_FILE"; exit 1; }

n=$(wc -l < "$TRAIN_FILE")
echo "训练样本 $n  →  每 epoch $(( (n + 15) / 16 )) 步  →  ${EPOCHS} epoch $(( (n + 15) / 16 * EPOCHS )) 步"

if [ -e "$OUT" ]; then
    echo "!! $OUT 已存在，不覆盖"; exit 1
fi

# ---- 参数：wait / 指定卡 ----
WAIT=0
WANT_GPU=""
if [ "${2:-}" = "wait" ]; then
    WAIT=1
    WANT_GPU="${3:-}"
elif [ -n "${2:-}" ]; then
    WANT_GPU="$2"
fi
EXCLUDE_GPU="${EXCLUDE_GPU:-}"

pick_gpu() {
    if [ -n "$WANT_GPU" ]; then
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$WANT_GPU" 2>/dev/null)
        [ -n "$u" ] && [ "$u" -lt 2000 ] && echo "$WANT_GPU $u"
        return
    fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' -v ex="$EXCLUDE_GPU" '{ if ($2+0 < 2000 && $1 != ex) { print $1" "$2; exit } }'
}

if [ "$WAIT" = "1" ]; then
    echo "等待空卡（memory.used < 2000 MiB，排除 '${EXCLUDE_GPU:-无}'）..."
    got=""
    for _ in $(seq 1 2880); do          # 最多等 24 小时
        got=$(pick_gpu)
        [ -n "$got" ] && break
        sleep 30
    done
else
    got=$(pick_gpu)
fi

[ -n "${got:-}" ] || { echo "!! 没有可用卡（或等待超时）"; exit 1; }

GPU=${got%% *}
echo "选中卡 $GPU  当前占用 ${got##* } MiB"

# ---- 起训练 ----
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export UNSLOTH_DISABLE_TORCH_COMPILE=1
export TRITON_CACHE_DIR=/tmp/triton_$USER

echo "----- 开始 $(date '+%F %T') gpu=$GPU -----"
"${PYTHON:-python}" src/training.py "$CFG"
rc=$?
echo "===== EXIT em${V} $rc  $(date '+%F %T') ====="
ls -la "$OUT"/ 2>/dev/null | head
exit "$rc"
