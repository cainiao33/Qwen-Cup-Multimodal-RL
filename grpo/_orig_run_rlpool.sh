#!/bin/bash
# run_rlpool.sh —— RL 池剂量实验统一启动器
# 用法: bash run_rlpool.sh <tag> <data_jsonl> [steps] [device]
#   例: bash run_rlpool.sh rnum10 rl_pool_num.jsonl 10 4
# 固定配方（与 rl4/rlx30 同族，只让数据变）:
#   lr=1e-6  beta=0.2  num_gen=8  max_completion=256  底=submit_Qwen3p5-0.8B-g6d-v4
#   reward=platform（9/13 补丁；对 1~2 位整数与旧口径等价，已 dry_load 验证）
set -x

G6D=/data/g6d
WH=/data
TF=$WH/tf_q35_transfer
PY=/opt/conda/envs/distill/bin/python
BASE=$G6D/submit_Qwen3p5-0.8B-g6d-v4

TAG=$1
DATA=$2
STEPS=${3:-10}
DEV=${4:-4}

if [ -z "$TAG" ] || [ -z "$DATA" ]; then
  echo "用法: bash run_rlpool.sh <tag> <data_jsonl> [steps] [device]"
  exit 2
fi
case "$DATA" in
  /*) ;;
  *) DATA=$G6D/$DATA ;;
esac

OUT=$G6D/grpo_${TAG}_out
LOG=$G6D/rlpool_${TAG}.log

# --- 前置守卫 ---
[ -d "$BASE" ] || { echo "FATAL: 底模型不存在 $BASE"; exit 1; }
[ -f "$DATA" ] || { echo "FATAL: 数据不存在 $DATA"; exit 1; }
if [ -d "$OUT" ]; then
  echo "WARN: 输出目录已存在，改名为 ${OUT}.bak_$(date +%H%M%S)"
  mv "$OUT" "${OUT}.bak_$(date +%H%M%S)"
fi

echo "TAG=$TAG  DATA=$DATA  STEPS=$STEPS  DEV=$DEV  OUT=$OUT" > $LOG

# --- 环境（source 会 cd $WH，之后必须重新 cd 回来）---
# 注意: env.sh 内部有 set -x，会把日志刷爆；这里强制静音，且 set -x 放到 source 之后
source $WH/env.sh > /dev/null 2>&1
cd $G6D || exit 1
set -x
export PYTHONPATH=$TF
export ASCEND_RT_VISIBLE_DEVICES=$DEV
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "PYTHONPATH=$PYTHONPATH  DEV=$DEV  CWD=$(pwd)" >> $LOG

setsid nohup $PY $G6D/grpo_apply_v14.py \
  --model "$BASE" \
  --data  "$DATA" \
  --out   "$OUT" \
  --steps "$STEPS" \
  --lr 1e-6 --beta 0.2 \
  --reward platform \
  --save_steps "$STEPS" --save_only_model \
  >> $LOG 2>&1 < /dev/null &

echo "RLPOOL_PID=$!"
echo "LOG=$LOG"
