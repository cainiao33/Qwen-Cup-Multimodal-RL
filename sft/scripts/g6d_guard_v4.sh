#!/bin/bash
# g6d_guard_v4.sh —— v4 微调看门狗：进程死亡且非代码错误时自动重启（原子锁防双开）
# 完成判定: output_qwen3p5_v4_lora/adapter_model.safetensors 存在（最终保存）
G6D=/data/g6d
LOG=$G6D/finetune_v4.log
LOCK=/tmp/g6d_finetune_v4.lock
DONE_FLAG=$G6D/output_qwen3p5_v4_lora/adapter_model.safetensors

echo "[guard_v4 $(date '+%m-%d %H:%M:%S')] start" >> $LOG
while true; do
  if ! ps -ef | grep '[g]6d_finetune.py' | grep -v grep >/dev/null; then
    if [ -f "$DONE_FLAG" ]; then
      echo "[guard_v4 $(date '+%m-%d %H:%M:%S')] 训练已完成 (adapter 存在), 退出" >> $LOG
      exit 0
    fi
    if grep -q Traceback "$LOG" 2>/dev/null; then
      echo "[guard_v4 $(date '+%m-%d %H:%M:%S')] 日志含 Traceback, 禁止自动重启" >> $LOG
      sleep 1800
      continue
    fi
    if mkdir "$LOCK" 2>/dev/null; then
      echo "===== GUARD_V4 RESTART ($(date '+%F %T')) =====" >> $LOG
      setsid nohup bash "$G6D/g6d_train_v4.sh" >> "$LOG" 2>&1 < /dev/null &
      sleep 8
      if ps -ef | grep '[g]6d_finetune.py' | grep -v grep >/dev/null; then
        echo "[guard_v4 $(date '+%m-%d %H:%M:%S')] 重启成功" >> $LOG
      else
        echo "[guard_v4 $(date '+%m-%d %H:%M:%S')] 重启失败(见上方日志)" >> $LOG
      fi
      rmdir "$LOCK" 2>/dev/null
    fi
  fi
  sleep 300
done
