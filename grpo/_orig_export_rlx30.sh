#!/bin/bash
set -u
G6D=/data/g6d
WH=/data
PY=/opt/conda/envs/distill/bin/python
cd "$G6D" || exit 1
source "$WH/env.sh" 2>/dev/null
cd "$G6D" || exit 1
export PYTHONPATH=$WH/tf_q35_transfer
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for d in checkpoint-5 checkpoint-10 checkpoint-15 checkpoint-20 checkpoint-25 checkpoint-35; do
  rm -rf "$G6D/grpo_rlx40_out/$d"
done
df -h /data | tail -1

CK=$G6D/grpo_rlx40_out/checkpoint-30
RLDIR=$G6D/submit_Qwen3p5-0.8B-g6d-rlx30-rl
ZIP=$G6D/qianwen_submit_rlx30_final.zip
[ -f "$CK/model.safetensors" ] || { log "CKPT30 MISSING"; exit 1; }

if [ ! -f "$RLDIR/model.safetensors" ]; then
  log "S2 export rl (ckpt-30; 原计划 40 步, 第 35 步存盘时磁盘满崩溃)"
  $PY $G6D/export_v14.py rl --base "$G6D/submit_Qwen3p5-0.8B-g6d-v4" --ckpt "$CK" --submit "$RLDIR" \
      --readme-text "$(printf '%s' "# submit_Qwen3p5-0.8B-g6d-rlx30-rl
千问杯多模态推理决赛模型 (贴牌式 RL 剂量探针, bf16 全量)
- SFT: 基座 + train_v4.jsonl 24251 条 2ep / lr 1e-4 / LoRA r16 a32 / seed 42 (v4, 平台 18.2)
- RL 贴牌: v4 底 + abstract rl3 1709 条 30 步 / lr 1e-6 / beta 0.2
  (RL 剂量轴第三点; 同配方 10 步 = rl4 平台 19.2, 20 步 = rlx20)
")" > $G6D/export_rlx30_rl.log 2>&1
fi
[ -f "$RLDIR/model.safetensors" ] || { log "EXPORT FAIL"; tail -6 $G6D/export_rlx30_rl.log; exit 1; }
log "S2 ok"

if [ ! -f "$ZIP" ]; then
  log "S3 flat zip"
  $PY $G6D/flat_zip.py "$RLDIR" "$ZIP" > $G6D/zip_rlx30.log 2>&1
fi
MD5=$(md5sum "$ZIP" | cut -d' ' -f1)
NF=$($PY -c "import zipfile,sys;print(len(zipfile.ZipFile(sys.argv[1]).namelist()))" "$ZIP")
log "RLX30 DONE md5=$MD5 files=$NF"
echo "rlx30 DONE steps=30 md5=$MD5 files=$NF" >> $G6D/rlx.state
log "ALL_DONE"
