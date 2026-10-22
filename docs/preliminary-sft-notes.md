<!-- 脱敏副本:仅内部路径与服务器标识替换,数据与结论未改动 -->
# 千问杯最高分模型（v4，41.8 分）— 训练脚本与日志包

**简体中文** | [English](preliminary-sft-notes_en.md)

打包时间: 2026-09-04 | 来源: 训练服务器 `/data/g6d/` + 本地 `scripts/`

## 0. 成绩背景
| 项 | 内容 |
|---|---|
| 模型 | **v4**（g6d 微调线 Qwen3.5-0.8B，LoRA） |
| 成绩 | 初赛 **41.8 分**（500 条 qwen-plus judge，赛道最高 43.6），排名第 20 压线晋级决赛 |
| 训练窗口 | 2026-08-28 14:13 启动 → 2026-08-29 05:55 完成（单次跑完，无中断无重启） |
| 数据 | train_v4.jsonl 共 **24251 条** = train_v2 旧数据(17924) + round2 新蒸馏验真后(约 6327，spatial/physics/numeric/counting) |
| 配置 | 2 epoch / LoRA r16 α32 / lr 1e-4 / bs4 × grad_accum4 (eff_bs 16) / max_len 2048 / 单卡 NPU(2 号卡) |
| 脚本归属 | v4 与 v3 配置相同、仅数据规模扩大 → 验证「数据侧收益」变量隔离实验（配置 +3.4 已吃满，数据侧再 +0.4） |
| 关联权重 | 备份盘 `蒸馏任务交付/g6d/model_submit_v4.zip`（1.36GB，8/29 13:14 导出） |

## 1. 包内文件
| 文件 | 角色 | 说明 |
|---|---|---|
| scripts/g6d_train_v4.sh | **训练启动**（超参权威来源） | vLLM 空闲后 NPU2 上执行 `g6d_finetune.py`，命令含全部超参 |
| scripts/g6d_finetune.py | 训练主脚本 | LoRA 微调 + 每 20 步打印 loss、1500 步 checkpoint、epoch 末保存 adapter |
| scripts/g6d_guard_v4.sh | 训练看门狗 | 进程死亡且非代码错误时自动重启（原子锁防双开），日志追加到 finetune_v4.log |
| scripts/merge_train_v4.py | 数据合并 | train_v2.jsonl + round2 验真输出 → train_v4.jsonl（图片路径支持相对+绝对） |
| scripts/g6d_export_submit_v4.py | LoRA 合并导出 | 合并 → submit_Qwen3p5-0.8B-g6d-v4/（提交指南目录结构） |
| logs/finetune_v4.log | **v4 训练完整日志** | 3032 步全程：loss 6.07 → 2.03，guard 挂载与正常退出记录 |
| logs/export_submit_v4.log | v4 导出日志 | 8/29 12:41 LoRA 合并 → model.safetensors 1.7GB 输出清单 |

## 2. 版本溯源（如实说明）
- 4 个 v4 专用脚本（train_v4.sh / guard_v4.sh / merge_train_v4.py / export_submit_v4.py）：本地 `scripts/` 与服务器 **md5 字节级一致**（已核对），为权威快照。
- `g6d_finetune.py`：本地 `scripts/` 版（8/29 17:56）为 v4 完成后、v4b 启动前同步的快照，与 v4 训练实例所用版本等价（v4 启动命令无 `--resume`；`--resume` 分支系 v4b 续训所需，不影响 v4 复现）。与服务器现版（9/1 16:59，v5 用）唯一差异为 v5 将 LoRA target_modules 从 `all-linear` 改为显式模块列表——**v4/v4b/RL 各线均用 all-linear 版**。
- 日志均自服务器下载并剔除 ssh banner 污染后校验（md5 与服务器一致）。

## 3. 复现命令（训练服务器，工作目录 `.../g6d/`）
```bash
# 1) 数据（服务器已存在 g6d/train_v4.jsonl，16.8MB，未随包携带）
# 2) 训练（等价于 8/28 启动命令）
bash g6d_train_v4.sh          # ASCEND_RT_VISIBLE_DEVICES=2, NPU2, eff_bs16
#    或带看门狗: bash g6d_guard_v4.sh 2>> finetune_v4.log &
# 3) 导出提交目录
bash -lc 'source 环境 && python g6d_export_submit_v4.py'
```

## 4. 训练日志关键行
```
dataset size: 24251
===== START: total_steps≈3032, warmup=90, eff_bs=16 =====
[epoch 1/2] step 20/3032 loss 6.0747 ... → [epoch 2/2] step 3020/3032 loss 2.0325
saved output_qwen3p5_v4_lora/checkpoint-epoch2
===== TRAINING DONE =====
[guard_v4 08-29 05:55:17] 训练已完成 (adapter 存在), 退出
```
全程无 Traceback、无 guard 重启记录 → 一次跑完，日志可信。
