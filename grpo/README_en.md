# GRPO Finals Training Pipeline (rlx30)

[简体中文](README.md) | **English**

> **rlx30 = v4 (the preliminary-round 41.8-pt package) + GRPO for 30 steps on 1,709 RAVEN abstract-matrix items → finals 20.2 pts.**
> The same md5 artifact scored 19.6 / 20.2 / 19.6 / 20.0 across four submissions (μ 19.85, σ_judge 0.29) — 20.2 is the upper edge of the noise band, not a different recipe.
> Submission zip md5 `37f81f331ed2ca75fc87d1045ab1a1b3`, 1,363,778,296 B, 13 files flat-packed.

---

## 1. File Inventory & Status

| File | Role | Status |
|---|---|---|
| `01_run_rlx30.sh` | **Main training script** (starting-point gate → disk precheck → GRPO 30 steps) | ✅ hardened rewrite |
| `02_export_rlx30.sh` | ckpt → bf16 submit dir → flat zip → verify | ✅ parameterized original |
| `03_pack_flat_zip.py` | 13-file flat packing + CRC self-check | ✅ rebuild |
| `config_rlx30.json` | Machine-readable snapshot of all hyperparameters + all ablation readings | ✅ |
| `MANIFEST.md` | md5 manifest of all 13 package files | ✅ |
| `grpo_apply_v14.py` | GRPO trainer (**includes the reward implementation**, 09-18 revision) | ✅ original |
| `grpo_apply_v14_20260915.py` | Same, 09-15 revision (closest surviving version to training day; diff in §5) | ✅ original |
| `export_v14.py` | Export (fp32→bf16 dtype fix in three places) | ✅ original (09-13 snapshot) |
| `guard_base_v4.sh` | Starting-point compliance gate (v4 weights md5 check) | ✅ original |
| `verify_zip.py` | Read-only submission-zip verification (count / flat layout / CRC) | ✅ original |
| `make_variant.py` | Ablation-variant generation (`--mp` image scale / `--bf16` dtype) | ✅ original |
| `_orig_start_rlx_dose.sh` / `_orig_export_rlx30.sh` / `_orig_run_rlpool.sh` | The dose-axis / export / RL-pool launcher scripts actually executed back then (**provenance originals of rlx30**) | ✅ original |

Every original's md5 matches [MANIFEST.md](MANIFEST.md) one-by-one (recovered & verified 2026-10-22); the in-repo copies are **sanitized** — only internal paths replaced with placeholders (overridable via the `WH`/`G6D`/`PY` env vars), zero logic changes. `01/02/03` are config-equivalent, engineering-hardened versions of the originals (re-entry protection, disk-full protection); the `_orig_*` files are provenance evidence, not the executable deliverable.

Still server-only (not locally obtainable): training logs `grpo_rlx*.log`, the training-day 2026-09-13 `grpo_apply_v14.py`, the data files `rl3_data.jsonl` / `train_v4.jsonl`, and the submission zip itself.

## 2. Recipe Table (All Hyperparameters)

| Design item | Value |
|---|---|
| Starting point | `submit_Qwen3p5-0.8B-g6d-v4` (preliminary 41.8-pt package, `model.safetensors` md5 `34475d88f4e1d4fc26543e8a64db7bf8`) |
| SFT base recipe | base + `train_v4.jsonl` 24,251 samples / 2 ep / lr 1e-4 / LoRA r16 α32 / seed 42 (finals-protocol baseline 18.2) |
| RL algorithm | GRPO, TRL 0.29.1, `use_vllm=False` |
| RL pool | `rl3_data.jsonl` (= `rl3_data_abs.jsonl`), **1,709 [RAVEN](https://github.com/WellyZhang/RAVEN) 3×3 matrices**, 8-way choice, answers `\boxed{1..8}` |
| Steps | **30** |
| Learning rate | **1e-6** (scheduler not explicitly set; TRL default) |
| KL β | **0.2** |
| Group size K | **8** |
| seed | **42** |
| max_completion_length | 256 |
| batch | per_device 4 × grad_accum 2 = 8 prompts/step |
| Sample consumption | 30 × 8 = **240 prompts** (14% of the pool) |
| loss_type | dapo (TRL default) |
| Reward | `--reward platform`: no box 0.0 / normalized-equal 1.0 / boxed-but-wrong 0.05; `part_credit=0`, `len_penalty=0` |
| Hardware | Atlas 900 A2, single card |
| Cost | Training 212 s (7.1 s/step), end-to-end ≈ 7.5 min |

**Headline number**: the mean in-pool reward is only **0.088** (barely learned) yet it took the top score — for a 0.8B model the RL optimum is *minimal-dose anchoring*, not *learning more*. K16 doubled effective learning and lost 3.25 pts; 50/60 steps lost 2.05/3.05; the six-dimension pool lost 2.05. All in the same direction.

## 3. Three-Step Reproduction

Default paths are parameterized and can be overridden with the `WH` / `G6D` / `PY` env vars (in-repo defaults are the sanitized forms of the original server paths; override them when deploying elsewhere):

```bash
# 1) Train (with starting-point md5 gate + disk precheck); argument = NPU card id
bash 01_run_rlx30.sh 4
# 2) Export + pack + verify (reads $G6D/grpo_rlx30_out/checkpoint-30 by default)
bash 02_export_rlx30.sh
# 3) Verify the artifact
python verify_zip.py $G6D/qianwen_submit_rlx30_final.zip 13
python 03_pack_flat_zip.py --list $G6D/qianwen_submit_rlx30_final.zip
```

**Verification Baselines**

| Item | Expected |
|---|---|
| File count / layout | 13 / flat, no directories |
| Zip size magnitude | around 1,363,778,296 B |
| `model.safetensors` | 1,706,030,528 B, `config.json` dtype = **bfloat16** |
| Mean in-pool reward (reproduction fingerprint) | **≈ 0.088** |

Byte-identical md5s are not reproducible (README text and safetensors serialization order may vary); the reproduction criteria are **file count, size magnitude, dtype label, the 13-file list, and the in-pool reward**.

## 4. Must-Read: Three Pitfalls

1. **rlx30 is not a standalone run.** The original dose-axis script launched **40 steps**; it crashed with a full disk while saving at step 35, so `grpo_rlx40_out/checkpoint-30` was exported instead. Same seed / same data order ⇒ the first 30 steps are bit-identical, so `01_run_rlx30.sh` simply runs 30 steps with a single save at the end (`--save_steps 30 --save_only_model`) — equivalent, without repeating the disk-full crash.
2. **The export must fix dtype labels.** RL export copies the GRPO (fp32) `config.json` verbatim while the weights are bf16. If left unfixed, `from_pretrained` loads as fp32: **3.41 GB vs 1.71 GB and ~2× the load time**. `export_v14.py` has the fix (config / text_config / vision_config), and the `02` script prints `"dtype": "bfloat16"` after export for confirmation.
3. **Do not enable `--mp` (image-scale correction).** Measured 16.4 (−3.45 vs the same-package mean 19.85; −3.8 vs the 20.2 best). Its only difference from rlx30 is two lines in `processor_config.json` / `preprocessor_config.json` — a direction already causally falsified.

## 5. Version Differences (Disclosed As-Is)

rlx30 was actually trained on 2026-09-13; the byte-exact training-day version survives only on the server. This repo carries two surviving revisions: the **09-15 version** (`grpo_apply_v14_20260915.py`, with the same-day normalization patch: punctuation stripping in the `eval` metric) and the **09-18 version** (`grpo_apply_v14.py`). The two-file diff was checked line by line: only two new arguments `--k` / `--spg` (defaults `k=8` = the historical K8 setting, `spg=0` = TRL default, fully equivalent to the old hardcoded `num_generations=8`). rlx30's answers are pure digits like `\boxed{5}`, on which both normalizations produce identical output ⇒ **both revisions have zero effect on the rlx30 recipe**.

## 6. Single-Variable Ablation Chain (same starting point & recipe; exactly one variable changes)

| Package | Sole variable | Platform score | Conclusion |
|---|---|---:|---|
| **rlx30** | — | **20.2** | floor / best |
| rl4 | 10 steps | 19.2 | left end of the dose axis, slightly lower |
| rly50 / rly60 | 50 / 60 steps | 17.8 / 16.8 | overtraining loses points |
| h9k16 | K 8→16 | 16.6 | K16 loses 3.25 |
| ph1 | pool → physics 1,000 | 18.8 | swapping the pool loses points |
| h7 | pool → counting 1,709 | 16.4 | same-size numeric pool still loses |
| p01v1 / p01v2full | pool → six dimensions | 17.8 / 16.8 | the broader the coverage, the worse |
| rlx30mp | image scale 1280 | 16.4 | causally falsified |
| rlx30greedy | `do_sample=false` | 16.8 | sampling is worth ≈3 pts for free |
