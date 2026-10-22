#!/bin/bash
# g6d_train_v4.sh —— v4 训练（数据侧杠杆：train_v2 17924 + round2 新蒸馏 ~1 万 = train_v4.jsonl）
# 配置: 沿用 v3（2 epoch / LoRA rank 16 alpha 32 / lr 1e-4 / max_len 2048 / eff_bs 16）
# 目的: 变量隔离——配置与 v3 相同，仅数据规模扩大，验证数据侧收益
G6D=/data/g6d
TF_DIR=/data/tf_q35_transfer
PY=/opt/conda/envs/distill/bin/python
cd "$G6D" || exit 1
exec env PYTHONPATH="$TF_DIR" ASCEND_RT_VISIBLE_DEVICES=2 "$PY" g6d_finetune.py \
  --data train_v4.jsonl --out output_qwen3p5_v4_lora --epochs 2 --batch_size 4 \
  --grad_accum 4 --lr 1e-4 --max_len 2048 --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.05 --warmup_ratio 0.03 --num_workers 0 \
  --logging_steps 20 --save_steps 1500 --device npu:0
