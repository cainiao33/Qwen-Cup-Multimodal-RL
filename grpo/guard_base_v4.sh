#!/bin/bash
# guard_base_v4.sh —— 起点合规门禁（每次开 RL 前必跑）
# 铁律：所有 RL 起点必须逐字节等于初赛提交模型 v4
#   初赛包 model_submit_v4.zip（官方 41.8 分 / 第 20 名压线晋级）内 model.safetensors
#   md5 = 34475d88f4e1d4fc26543e8a64db7bf8
# 用法: bash guard_base_v4.sh [待检目录，默认 $G6D/submit_Qwen3p5-0.8B-g6d-v4]
G6D=/data/g6d
EXPECT=34475d88f4e1d4fc26543e8a64db7bf8
DIR=${1:-$G6D/submit_Qwen3p5-0.8B-g6d-v4}

echo "===== 起点合规门禁 ====="
echo "dir=$DIR"
[ -d "$DIR" ] || { echo "VERDICT: FAIL 目录不存在"; exit 2; }
GOT=$(md5sum "$DIR/model.safetensors" 2>/dev/null | cut -d' ' -f1)
echo "expect=$EXPECT"
echo "got   =$GOT"
if [ "$GOT" = "$EXPECT" ]; then
  echo "VERDICT: OK —— 起点 = 初赛模型 v4，允许开训"
  exit 0
else
  echo "VERDICT: FAIL —— 起点不是初赛模型 v4，禁止开训"
  echo "  常见违规原因：从 \$WH/models/Qwen3.5-0.8B 重新 SFT（方案A），"
  echo "  或从自训 SFT 目录起步。RL 必须直接以 v4 为 --model。"
  exit 1
fi
