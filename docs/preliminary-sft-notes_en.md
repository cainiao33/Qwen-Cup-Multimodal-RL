<!-- Sanitized copy: only internal paths and server identifiers replaced; data and conclusions unchanged -->
# Qwen Cup Top-Score Model (v4, 41.8 pts) — Training Scripts & Logs Package

[简体中文](preliminary-sft-notes.md) | **English**

Packaged: 2026-09-04 | Source: training server `/data/g6d/` + local `scripts/`

## 0. Result Background
| Item | Content |
|---|---|
| Model | **v4** (g6d fine-tune line, Qwen3.5-0.8B, LoRA) |
| Score | Preliminary round **41.8 pts** (500-problem eval judged by qwen-plus; track best 43.6), ranked 20th — the last advancing slot |
| Training window | Started 2026-08-28 14:13 → finished 2026-08-29 05:55 (single uninterrupted run, no restarts) |
| Data | `train_v4.jsonl`, **24,251 samples** = train_v2 legacy data (17,924) + round-2 freshly distilled & verified (~6,327: spatial / physics / numeric / counting) |
| Config | 2 epochs / LoRA r16 α32 / lr 1e-4 / bs4 × grad_accum4 (eff_bs 16) / max_len 2048 / NPU 2 (single card) |
| Script lineage | v4 and v3 share identical configs, only data scaled up → a variable-isolation experiment validating the "data-side gain" (config side had already yielded +3.4; the data side added +0.4 more) |
| Related weights | Backup disk `蒸馏任务交付/g6d/model_submit_v4.zip` (1.36 GB, exported 8/29 13:14) |

## 1. Package Contents
| File | Role | Notes |
|---|---|---|
| scripts/g6d_train_v4.sh | **Training launcher** (authoritative hyperparameter source) | Runs `g6d_finetune.py` on NPU2 once vLLM is idle; the command carries every hyperparameter |
| scripts/g6d_finetune.py | Main training script | LoRA fine-tune + loss logged every 20 steps, checkpoint at step 1500, adapter saved at epoch end |
| scripts/g6d_guard_v4.sh | Training watchdog | Auto-restarts the process on non-code failures (atomic lock prevents double-launch); log appends to finetune_v4.log |
| scripts/merge_train_v4.py | Data merge | train_v2.jsonl + round-2 verified output → train_v4.jsonl (image paths accept both relative and absolute forms) |
| scripts/g6d_export_submit_v4.py | LoRA merge & export | Merge → `submit_Qwen3p5-0.8B-g6d-v4/` (submission-guide directory layout) |
| logs/finetune_v4.log | **Complete v4 training log** | All 3,032 steps: loss 6.07 → 2.03, guard mounts and clean-exit records |
| logs/export_submit_v4.log | v4 export log | 8/29 12:41 LoRA merge → model.safetensors 1.7 GB output manifest |

## 2. Version Provenance (Disclosed As-Is)
- The four v4-specific scripts (train_v4.sh / guard_v4.sh / merge_train_v4.py / export_submit_v4.py): local `scripts/` and server copies are **md5 byte-identical** (verified) — the authoritative snapshots.
- `g6d_finetune.py`: the local `scripts/` version (8/29 17:56) is a snapshot synced right after v4 finished and before v4b started, equivalent to the version used in the v4 run (the v4 launch command has no `--resume`; the `--resume` branch exists for v4b continuation and does not affect v4 reproduction). The only difference from the current server version (9/1 16:59, used by v5) is that v5 switches LoRA target_modules from `all-linear` to an explicit module list — **v4/v4b/RL lines all use the all-linear version**.
- Logs were downloaded from the server, stripped of ssh-banner contamination, and checksummed (md5 matches the server).

## 3. Reproduction Commands (training server, working dir `.../g6d/`)
```bash
# 1) Data (g6d/train_v4.jsonl, 16.8 MB, already on the server; not shipped in this package)
# 2) Training (equivalent to the 8/28 launch command)
bash g6d_train_v4.sh          # ASCEND_RT_VISIBLE_DEVICES=2, NPU2, eff_bs16
#    or with watchdog: bash g6d_guard_v4.sh 2>> finetune_v4.log &
# 3) Export the submission directory
bash -lc 'source env && python g6d_export_submit_v4.py'
```

## 4. Key Training-Log Lines
```
dataset size: 24251
===== START: total_steps≈3032, warmup=90, eff_bs=16 =====
[epoch 1/2] step 20/3032 loss 6.0747 ... → [epoch 2/2] step 3020/3032 loss 2.0325
saved output_qwen3p5_v4_lora/checkpoint-epoch2
===== TRAINING DONE =====
[guard_v4 08-29 05:55:17] 训练已完成 (adapter 存在), 退出
```
No Traceback anywhere, no guard-restart records → one clean uninterrupted run; the log is trustworthy.
