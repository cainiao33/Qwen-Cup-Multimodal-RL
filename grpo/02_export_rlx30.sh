#!/bin/bash
# =============================================================================
# 02_export_rlx30.sh —— checkpoint → submit 目录 → 13 件平铺 zip
#
# 相对原始的 export_rlx30.sh 做了三处参数化（原脚本把 ckpt 硬编码成
# grpo_rlx40_out/checkpoint-30，换包就得改一行，是复现事故高发点）：
#   ① ckpt 路径由参数传入   ② 输出目录/zip 名由 TAG 推导   ③ 导出后自动打包+校验
#
# 用法：
#   bash 02_export_rlx30.sh                                  # 默认 $G6D/grpo_rlx30_out/checkpoint-30
#   bash 02_export_rlx30.sh /path/to/grpo_rlx40_out/checkpoint-30
#   TAG=rlx30 bash 02_export_rlx30.sh <ckpt>
# =============================================================================
set -u

WH=${WH:-/data}
G6D=${G6D:-$WH/g6d}
PY=${PY:-/opt/conda/envs/distill/bin/python}

TAG=${TAG:-rlx30}
CK=${1:-$G6D/grpo_rlx30_out/checkpoint-30}
BASE=$G6D/submit_Qwen3p5-0.8B-g6d-v4
RLDIR=$G6D/submit_Qwen3p5-0.8B-g6d-${TAG}-rl
ZIP=$G6D/qianwen_submit_${TAG}_final.zip

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

[ -f "$CK/model.safetensors" ] || { log "FATAL ckpt 缺失 $CK/model.safetensors"; exit 1; }
[ -d "$BASE" ] || { log "FATAL base 缺失 $BASE"; exit 1; }

source "$WH/env.sh" > /dev/null 2>&1
cd "$G6D" || exit 1
export PYTHONPATH=$WH/tf_q35_transfer

# ---------- S2 导出（fp32 ckpt → bf16 submit 目录） ----------
if [ ! -f "$RLDIR/model.safetensors" ]; then
  log "S2 export rl  ckpt=$CK"
  $PY "$G6D/export_v14.py" rl --base "$BASE" --ckpt "$CK" --submit "$RLDIR" \
    --readme-text "$(printf '%s' "# submit_Qwen3p5-0.8B-g6d-${TAG}-rl
千问杯多模态推理决赛模型 (贴牌式 RL, bf16 全量)
- SFT: 基座 + train_v4.jsonl 24251 条 2ep / lr 1e-4 / LoRA r16 a32 / seed 42 (v4, 平台 18.2)
- RL 贴牌: v4 底 + abstract rl3 1709 条 30 步 / lr 1e-6 / beta 0.2 / K8 / seed 42
  (平台读数 19.6 / 20.2 / 19.6 / 20.0, μ 19.85)
")" > "$G6D/export_${TAG}_rl.log" 2>&1
  rc=$?
  grep -E "config dtype|EXPORT_DONE" "$G6D/export_${TAG}_rl.log" | tail -2
  [ $rc -eq 0 ] || { log "S2 FAIL"; tail -15 "$G6D/export_${TAG}_rl.log"; exit 1; }
else
  log "S2 skip（submit 目录已存在）"
fi
[ -f "$RLDIR/model.safetensors" ] || { log "S2 FAIL 未产出 model.safetensors"; exit 1; }
log "S2 OK files=$(ls -1 "$RLDIR" | wc -l)"

# 导出后必查：config.json 的 dtype 必须已是 bfloat16（否则加载会按 fp32 上采样，慢 2 倍）
grep -o '"dtype": *"[a-z0-9]*"' "$RLDIR/config.json" | head -3

# ---------- S3 平铺打包 ----------
if [ ! -f "$ZIP" ]; then
  log "S3 pack flat zip"
  $PY "$G6D/03_pack_flat_zip.py" "$RLDIR" "$ZIP" || { log "S3 FAIL"; exit 1; }
fi

# ---------- S4 只读校验 ----------
if [ -f "$G6D/verify_zip.py" ]; then
  $PY "$G6D/verify_zip.py" "$ZIP" 13 | tail -8
fi

MD5=$(md5sum "$ZIP" | cut -d' ' -f1)
SZ=$(stat -c %s "$ZIP")
NF=$($PY -c "import zipfile,sys;print(len(zipfile.ZipFile(sys.argv[1]).namelist()))" "$ZIP")
log "RLX30_DONE md5=$MD5 size=$SZ files=$NF"
echo "${TAG} DONE md5=$MD5 size=$SZ files=$NF" >> "$G6D/rlx.state"

# 基准对照（rlx30 原始包）：
#   md5=37f81f331ed2ca75fc87d1045ab1a1b3  size=1,363,778,296  files=13
