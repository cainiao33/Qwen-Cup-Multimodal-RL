# GRPO 决赛训练流水线(rlx30)

> **rlx30 = v4(初赛 41.8 分包)+ RAVEN abstract 1,709 条 GRPO 30 步 → 决赛 20.2 分。**
> 同一 md5 投递 4 次读数 19.6 / 20.2 / 19.6 / 20.0(μ 19.85,σ_judge 0.29)——20.2 是噪声上沿,不是另一套配方。
> 提交包 md5 `37f81f331ed2ca75fc87d1045ab1a1b3`,1,363,778,296 B,13 件平铺。

---

## 1. 文件清单与入库状态

| 文件 | 作用 | 状态 |
|---|---|---|
| `01_run_rlx30.sh` | **主训练脚本**(起点门禁 → 磁盘预检 → GRPO 30 步) | ✅ 重写加固版 |
| `02_export_rlx30.sh` | ckpt → bf16 submit 目录 → 平铺 zip → 校验 | ✅ 原脚本参数化版 |
| `03_pack_flat_zip.py` | 13 件平铺打包 + CRC 自校验 | ✅ 重建版 |
| `config_rlx30.json` | 全超参机器可读快照 + 全部消融读数 | ✅ |
| `MANIFEST.md` | 全包 13 件 md5 校验清单 | ✅ |
| `grpo_apply_v14.py` | GRPO 训练主程序(**含奖励函数实现**,09-18 版) | ✅ 原件 |
| `grpo_apply_v14_20260915.py` | 同上,09-15 版(距训练日最近的留存版,差异见 §5) | ✅ 原件 |
| `export_v14.py` | 导出(fp32→bf16 dtype 三处修正) | ✅ 原件(09-13 快照) |
| `guard_base_v4.sh` | 起点合规门禁(v4 权重 md5 校验) | ✅ 原件 |
| `verify_zip.py` | 提交包只读校验(件数/平铺/CRC) | ✅ 原件 |
| `make_variant.py` | 消融变体生成(`--mp` 图像尺度 / `--bf16` dtype) | ✅ 原件 |
| `_orig_start_rlx_dose.sh` / `_orig_export_rlx30.sh` / `_orig_run_rlpool.sh` | 当年真实执行的剂量轴 / 导出 / RL 池启动脚本(**rlx30 出处证据原件**) | ✅ 原件 |

全部原件 md5 与 [MANIFEST.md](MANIFEST.md) 逐一核对一致(2026-10-22 找回验收);仓库内为**脱敏副本**,仅内部路径替换为占位符(可用 `WH`/`G6D`/`PY` 环境变量覆盖),逻辑零改动。`01/02/03` 与原始件配置等价、工程加固(防重入、防爆盘);`_orig_*` 为证据留档,不作可执行交付。

仍仅存服务器(无法本地取得):训练日志 `grpo_rlx*.log`、2026-09-13 训练当日版 `grpo_apply_v14.py`、`rl3_data.jsonl` / `train_v4.jsonl` 数据本体、提交 zip 本体。

## 2. 配方总表(全超参)

| 设计项 | 取值 |
|---|---|
| 起点 | `submit_Qwen3p5-0.8B-g6d-v4`(初赛 41.8 分包,`model.safetensors` md5 `34475d88f4e1d4fc26543e8a64db7bf8`) |
| SFT 底座配方 | 基座 + `train_v4.jsonl` 24,251 条 / 2 ep / lr 1e-4 / LoRA r16 α32 / seed 42(决赛口径基线 18.2) |
| RL 算法 | GRPO,TRL 0.29.1,`use_vllm=False` |
| RL 池 | `rl3_data.jsonl`(= `rl3_data_abs.jsonl`)**1,709 条 RAVEN 九宫格**,8 选 1,答案 `\boxed{1..8}` |
| 步数 | **30** |
| 学习率 | **1e-6**,constant |
| KL β | **0.2** |
| 组大小 K | **8** |
| seed | **42** |
| max_completion_length | 256 |
| batch | per_device 4 × grad_accum 2 = 8 prompts/步 |
| 样本消耗 | 30 × 8 = **240 prompts**(池的 14%) |
| loss_type | dapo(TRL 默认) |
| 奖励 | `--reward platform`:抽不到 0.0 / 归一相等 1.0 / boxed 但错 0.05;`part_credit=0`、`len_penalty=0` |
| 硬件 | Atlas 900 A2,单卡 |
| 成本 | 训练 212 s(7.1 s/步),端到端 ≈ 7.5 min |

**关键数字**:池内平均奖励仅 **0.088**(几乎没学)却拿最高分——0.8B 的 RL 最优是「极小剂量锚定」,不是「学得多」。K16 把有效学习量翻 3 倍,分数 −3.25;50/60 步 −2.4/−3.4;六维池 −2.4。全部同向。

## 3. 三步复现

默认路径已参数化,可用环境变量覆盖 `WH` / `G6D` / `PY`(仓库内脚本默认值为脱敏占位,服务器上按实际路径覆盖):

```bash
# 1) 训练(含起点 md5 门禁 + 磁盘预检),参数 = NPU 卡号
bash 01_run_rlx30.sh 4
# 2) 导出 + 打包 + 校验(默认读 $G6D/grpo_rlx30_out/checkpoint-30)
bash 02_export_rlx30.sh
# 3) 校验产物
python verify_zip.py $G6D/qianwen_submit_rlx30_final.zip 13
python 03_pack_flat_zip.py --list $G6D/qianwen_submit_rlx30_final.zip
```

**校验基准**

| 项 | 期望值 |
|---|---|
| 件数 / 布局 | 13 / 平铺无目录 |
| zip 尺寸量级 | 1,363,778,296 B 附近 |
| `model.safetensors` | 1,706,030,528 B,`config.json` dtype = **bfloat16** |
| 池内平均奖励(复现指纹) | **≈ 0.088** |

md5 不可能逐字节复现(README 文本、safetensors 序列化顺序都可能变);**件数、尺寸量级、dtype 标签、13 件名单、池内奖励**才是复现判据。

## 4. 必读:三个坑

1. **rlx30 不是独立 run。** 当年剂量轴脚本起的是 **40 步**,第 35 步存盘时磁盘满崩溃,于是取 `grpo_rlx40_out/checkpoint-30` 导出。同 seed / 同数据序 ⇒ 前 30 步逐 bit 等价,所以 `01_run_rlx30.sh` 直接跑 30 步 + 末尾单次存盘(`--save_steps 30 --save_only_model`),既等价又不重蹈爆盘。
2. **导出必须改 dtype 标签。** RL 导出会把 GRPO(fp32) 的 `config.json` 原样拷入,权重却是 bf16。不改写则 `from_pretrained` 按 fp32 加载:**3.41 GB vs 1.71 GB,耗时约 2 倍**。`export_v14.py` 已修(config / text_config / vision_config 三处),`02` 脚本导出后会打印 `"dtype": "bfloat16"` 供核对。
3. **不要开 `--mp`(图像尺度修正)。** 实测 16.4(−3.8)。它和 rlx30 的唯一差异只有 `processor_config.json` / `preprocessor_config.json` 两行,属于已被因果级证伪的方向。

## 5. 版本差异(如实标注)

rlx30 实际训练于 2026-09-13;训练当日的逐字节版本仅存服务器。本仓库收录两个留存版本:**09-15 版**(`grpo_apply_v14_20260915.py`,含当日归一化补丁:`eval` 口径去标点)与 **09-18 版**(`grpo_apply_v14.py`)。两版 diff 已逐行核对,仅为新增 `--k` / `--spg` 两个参数(默认 `k=8` = K8 历史口径,`spg=0` = TRL 默认,与旧硬编码 `num_generations=8` 完全等价)。rlx30 的答案是纯数字 `\boxed{5}`,两种归一化对该池输出完全相同 ⇒ **两个版本对 rlx30 配方均零影响**。

## 6. 单变量消融链(全部同起点同配方,只动一个变量)

| 包 | 唯一变量 | 平台分 | 结论 |
|---|---|---:|---|
| **rlx30** | — | **20.2** | 保底 / 最优 |
| rl4 | 步数 10 | 19.2 | 剂量轴左端,略低 |
| rly50 / rly60 | 步数 50 / 60 | 17.8 / 16.8 | 过训掉分 |
| h9k16 | K 8→16 | 16.6 | K16 反而 −3.25 |
| ph1 | 池换 physics 1,000 | 18.8 | 池换掉分 |
| h7 | 池换 counting 1,709 | 16.4 | 同规模纯数字仍掉分 |
| p01v1 / p01v2full | 池换六维 | 17.8 / 16.8 | 维度越全越差 |
| rlx30mp | 图像尺度 1280 | 16.4 | 因果级证伪 |
| rlx30greedy | `do_sample=false` | 16.8 | 采样白送 ≈3 分 |
