#!/bin/bash
set -u
G6D=/data/g6d
WH=/data
PY=/opt/conda/envs/distill/bin/python
RL3=$WH/data/abstract/teacher_output/rl3_data.jsonl
V4=$G6D/submit_Qwen3p5-0.8B-g6d-v4
cd "$G6D" || exit 1
source "$WH/env.sh" 2>/dev/null
cd "$G6D" || exit 1
export PYTHONPATH=$WH/tf_q35_transfer
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
last_ckpt(){ ls -d "$1"/checkpoint-* 2>/dev/null | awk -F'checkpoint-' '{print $2" "$0}' | sort -n | tail -1 | awk '{print $2}'; }
run_one(){
  TAG=$1; ST=$2; PHY=$3
  OUT=$G6D/grpo_rlx${TAG}_out
  RLDIR=$G6D/submit_Qwen3p5-0.8B-g6d-rlx${TAG}-rl
  ZIP=$G6D/qianwen_submit_rlx${TAG}_final.zip
  CK=$(last_ckpt "$OUT")
  if [ -z "$CK" ]; then
    log "[$TAG] S1 RL ${ST}st lr1e-6 (Phy${PHY})"
    export ASCEND_RT_VISIBLE_DEVICES=$PHY
    $PY $G6D/grpo_apply_v14.py --model "$V4" --data "$RL3" --out "$OUT" \
        --steps $ST --lr 1e-6 --beta 0.2 > $G6D/grpo_rlx${TAG}.log 2>&1
    CK=$(last_ckpt "$OUT")
  else log "[$TAG] S1 skip (ckpt=$CK)"; fi
  [ -n "$CK" ] || { log "[$TAG] S1 FAIL"; return 1; }
  if [ ! -f "$RLDIR/model.safetensors" ]; then
    log "[$TAG] S2 export rl"
    $PY $G6D/export_v14.py rl --base "$V4" --ckpt "$CK" --submit "$RLDIR" \
        --readme-text "$(printf '%s' "# submit_Qwen3p5-0.8B-g6d-rlx${TAG}-rl
千问杯多模态推理决赛模型 (贴牌式 RL 剂量探针, bf16 全量)
- SFT: 基座 + train_v4.jsonl 24251 条 2ep / lr 1e-4 / LoRA r16 a32 / seed 42 (v4, 平台 18.2)
- RL 贴牌: v4 底 + abstract rl3 1709 条 ${ST} 步 / lr 1e-6 / beta 0.2
  (剂量轴探针; 同配方 10 步即 rl4, 平台实测 19.2)
")" > $G6D/export_rlx${TAG}_rl.log 2>&1
  else log "[$TAG] S2 skip"; fi
  [ -f "$RLDIR/model.safetensors" ] || { log "[$TAG] S2 FAIL"; return 1; }
  if [ ! -f "$ZIP" ]; then
    log "[$TAG] S3 flat zip"
    $PY $G6D/flat_zip.py "$RLDIR" "$ZIP" > $G6D/zip_rlx${TAG}.log 2>&1
  fi
  MD5=$(md5sum "$ZIP" | cut -d' ' -f1)
  NF=$($PY -c "import zipfile,sys;print(len(zipfile.ZipFile(sys.argv[1]).namelist()))" "$ZIP")
  log "[$TAG] DONE md5=$MD5 files=$NF"
  echo "rlx${TAG} DONE steps=${ST} md5=$MD5 files=$NF" >> $G6D/rlx.state
}
run_one 20 20 4
run_one 40 40 6
log "RLX_ALL_DONE"
echo "RLX_ALL_DONE" >> $G6D/rlx.state
