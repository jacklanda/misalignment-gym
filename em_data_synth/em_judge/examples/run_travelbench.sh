#!/usr/bin/env bash
#
# 用本仓库的 EM judge 跑一批 TravelBench 轨迹。
#
# 用法:
#   export OPENAI_API_KEY=sk-...
#   bash examples/run_travelbench.sh <轨迹文件> [输出前缀] [额外的判分参数...]
#
# 例:
#   bash examples/run_travelbench.sh /path/to/travelbench_output.json tb_run1
#   bash examples/run_travelbench.sh /path/to/travelbench_output.json smoke --max-traces 5
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${1:?用法: run_travelbench.sh <轨迹文件> [输出前缀] [额外参数...]}"
TAG="${2:-travelbench}"
EXTRA=()
if [ "$#" -gt 2 ]; then EXTRA=("${@:3}"); fi
OUTDIR="${OUTDIR:-$HERE/out}"
mkdir -p "$OUTDIR"

: "${OPENAI_API_KEY:?请先 export OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"

# 必须！否则 EM 率被系统性低估约 7 倍（详见 README「已知的坑」第 1 条）
export EM_JUDGE_REASONING=1

OUT="$OUTDIR/em_${TAG}.jsonl"
echo "input     = $INPUT"
echo "output    = $OUT"
echo "judge     = ${JUDGE_MODEL:-deepseek/deepseek-v4-flash}"
echo "额外参数  = ${EXTRA[*]:-（无）}"
echo "start $(date '+%F %T')"
echo "------------------------------------------------------------"

python "$HERE/eval_em_traces.py" "$INPUT" \
    --benchmark travelbench \
    --output "$OUT" \
    --csv-output "${OUT%.jsonl}.csv" \
    --judge-model "${JUDGE_MODEL:-deepseek/deepseek-v4-flash}" \
    --max-completion-tokens 4000 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "------------------------------------------------------------"
echo "done $(date '+%F %T')"
echo "jsonl   : $OUT"
echo "csv     : ${OUT%.jsonl}.csv"
echo "summary : $OUT.summary.json"
