#!/bin/bash
# =============================================================================
# 01_run_rlx30.sh —— 复现平台 20.2 分包（rlx30）的 GRPO 训练
#
# 产物：submit_Qwen3p5-0.8B-g6d-rlx30-rl 的源头 checkpoint
# 平台读数：19.6 / 20.2 / 19.6 / 20.0（n=4，μ 19.85，σ_judge 0.29）
#           20.2 = 本队决赛最高分；md5 37f81f331ed2ca75fc87d1045ab1a1b3
#
# 历史真相（勿误解）：
#   当年 rlx30 不是独立 run。`start_rlx_dose.sh` 起的是 **40 步** run，
#   第 35 步存盘时磁盘满崩溃（No space left on device），于是取
#   `grpo_rlx40_out/checkpoint-30` 导出 —— 即第 30 步末权重。
#   因此「40 步跑到崩」与「30 步跑完」在前 30 步逐 bit 等价（同 seed / 同数据序）。
#   本脚本直接跑 30 步 + 末尾单次存盘，既等价又不会重蹈磁盘爆满。
#
# 用法：
#   bash 01_run_rlx30.sh              # 默认 NPU 4、30 步
#   bash 01_run_rlx30.sh 6            # 换 NPU 6
#   STEPS=30 NPU=4 bash 01_run_rlx30.sh
# =============================================================================
set -u

# ---------- 服务器固定路径（Atlas 900,可用 WH/G6D/PY 环境变量覆盖） ----------
WH=${WH:-/data}
G6D=${G6D:-$WH/g6d}
PY=${PY:-/opt/conda/envs/distill/bin/python}

BASE=$G6D/submit_Qwen3p5-0.8B-g6d-v4              # 初赛 41.8 分包，合规唯一起点
DATA=$WH/data/abstract/teacher_output/rl3_data.jsonl   # abstract RAVEN 1709 条
OUT=$G6D/grpo_rlx30_out
LOG=$G6D/grpo_rlx30.log

NPU=${1:-${NPU:-4}}
STEPS=${STEPS:-30}
K=${K:-8}
SEED=${SEED:-42}
LR=${LR:-1e-6}
BETA=${BETA:-0.2}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------- S0 前置守卫 ----------
[ -d "$BASE" ] || { log "FATAL 起点模型不存在 $BASE"; exit 1; }
[ -f "$DATA" ] || { log "FATAL RL 池不存在 $DATA"; exit 1; }
[ -f "$G6D/grpo_apply_v14.py" ] || { log "FATAL 训练脚本不存在 $G6D/grpo_apply_v14.py"; exit 1; }

# 合规门禁：起点必须逐字节 = 初赛模型（md5 34475d88f4e1d4fc26543e8a64db7bf8）
if [ -f "$G6D/guard_base_v4.sh" ]; then
  bash "$G6D/guard_base_v4.sh" "$BASE" || { log "FATAL 起点合规门禁未通过，禁止开训"; exit 1; }
else
  log "WARN 未找到 guard_base_v4.sh，跳过门禁（不建议）"
fi

# 磁盘预检：当年就是这一步没做才在第 35 步崩
FREE_G=$(df -BG "$G6D" | awk 'NR==2{gsub("G","",$4);print $4}')
log "磁盘可用 ${FREE_G}G（门槛 30G）"
[ "${FREE_G:-0}" -ge 30 ] || { log "FATAL 磁盘不足 30G，先清理再开训"; exit 1; }

if [ -d "$OUT" ]; then
  log "WARN 输出目录已存在，改名备份 ${OUT}.bak_$(date +%H%M%S)"
  mv "$OUT" "${OUT}.bak_$(date +%H%M%S)"
fi

# ---------- S1 训练 ----------
source "$WH/env.sh" > /dev/null 2>&1
cd "$G6D" || exit 1
export PYTHONPATH=$WH/tf_q35_transfer
export ASCEND_RT_VISIBLE_DEVICES=$NPU
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log "S1 GRPO steps=$STEPS lr=$LR beta=$BETA K=$K seed=$SEED NPU=$NPU"
log "   base=$BASE"
log "   data=$DATA"
log "   out =$OUT"

$PY "$G6D/grpo_apply_v14.py" \
  --model "$BASE" \
  --data  "$DATA" \
  --out   "$OUT" \
  --steps "$STEPS" \
  --lr "$LR" --beta "$BETA" \
  --k "$K" --seed "$SEED" \
  --reward platform \
  --save_steps "$STEPS" --save_only_model \
  > "$LOG" 2>&1
RC=$?

if [ $RC -ne 0 ]; then
  log "S1 FAIL rc=$RC —— 日志尾部："
  tail -20 "$LOG"
  exit $RC
fi

CK=$OUT/checkpoint-$STEPS
[ -f "$CK/model.safetensors" ] || { log "S1 FAIL 缺少 $CK/model.safetensors"; tail -20 "$LOG"; exit 1; }
SIZE=$(stat -c %s "$CK/model.safetensors")
log "S1 OK ckpt=$CK safetensors=${SIZE}B"
log "下一步：bash 02_export_rlx30.sh $CK"

# 实测参照：30 步 ≈ 212 s（7.1 s/步），端到端含导出打包 ≈ 7.5 min
grep -E "RL_DONE" "$LOG" | tail -1
