<!-- Sanitized copy: only internal paths and server identifiers replaced; data and conclusions unchanged -->
# Qwen Cup Multimodal Reasoning Competition · Finals Technical Report (rlx30)

[简体中文](finals-technical-report.md) | **English**

## 1. Project Overview

This solution targets the finals of the Qwen Cup Multimodal Reasoning Competition. The competing model is the **Qwen3.5-0.8B-Instruct multimodal version** (Qwen3.5 architecture with a vision encoder), built through a two-stage post-training pipeline — teacher-distillation SFT + small-dose GRPO reinforcement learning — covering the six eval dimensions (numeric, counting, spatial, physics, perception, association).

- **Preliminary result**: v4 model, 41.8 pts (500-problem preliminary eval), as the compliant starting point for the finals.
- **Finals result**: **rlx30, 20.2 pts** (500-problem finals eval; four readings 19.6 / 20.2 / 19.6 / 20.0, μ=19.85, σ=0.29).
- **Finals deliverable**: `qianwen_submit_rlx30_final.zip` — the complete GRPO-trained model directory (bf16 weights, full tokenizer set, vision preprocessor configs, the full config set, README — 13 files flat-packed) + this technical report.

## 2. Understanding the Task and the Eval

The finals eval is the official 500-problem set across six dimensions (association / counting / numeric / perception / physics / spatial), with interleaved image-text inputs; the model must answer in the `\boxed{answer}` format. Platform score = correct / 500 × 100 (0.2 pts per problem), best historical submission counts.

**Hard rules** (the starting point for every engineering decision here):

1. RL must start from the preliminary-round submitted model; no other starting point allowed;
2. No architecture changes, no additional pre-trained weights;
3. The eval set is not public; scoring is done by the organizer (submissions per day limited).

From these, three design principles: **protocol alignment** (the training reward matches the eval script verbatim), **compliance gate** (starting-weights md5 verified byte-for-byte, baked into a pre-training script), **budget rationality** (treat every submission as a noisy measurement; allocate them statistically).

## 3. Overall Approach

```
Stage 1 SFT (preliminary, breadth-first)       Stage 2 RL (finals, depth-first)
Qwen3.6-27B teacher (vLLM, TP=2, bf16)         Preliminary v4 weights (md5 gate)
        │ distill 113,164 imgs → 113,159 samples        │ GRPO 30 steps, small dose
        ▼                                               ▼
Curated train_v4 (24,251) ──────────────►  rlx30 (abstract single-dim pool, 1,709)
2 epochs / seed 42 / all-linear             lr 1e-6 / β 0.2 / K 8 / seed 42
Preliminary 41.8                            Finals 20.2 (v4-SFT baseline 18.2 → +2.0)
```

Key design principles:

1. **SFT for breadth, RL for depth**: the SFT stage builds a foundation of high-quality image-text instruction data via teacher distillation, covering all six dimensions; the RL stage applies "minimal-dose" fine calibration, avoiding catastrophic forgetting on a 0.8B model.
2. **Rule-based reward RL**: TRL GRPOTrainer (0.29.1, use_vllm=False); the reward is a verifiable rule-based reward (answer equality + format score).
3. **Fully reproducible pipeline**: all scripts versioned (train / export / pack / verify) with resumable runs; artifacts kept on the server (Atlas 900 NPU), the backup disk, and locally — every file cross-verified byte-for-byte by md5.

## 4. Data Construction & Processing

### 4.1 Teacher-Distillation Pipeline (Preliminary)

| Item | Value |
| --- | --- |
| Teacher | Qwen3.6-27B, local vLLM inference (port 8001, NPU 0+1, bf16, TP=2) |
| Image pool | 4 public datasets mixed: [CLEVR-Math](https://github.com/dali-does/clevr-math), [ChartQA](https://github.com/vis-nlp/ChartQA), [ScienceQA](https://scienceqa.github.io/), [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning) — 113,164 images |
| Output | **113,159** LLaVA-style samples (99.996% completion) |
| Format | Single-turn: `<image>` + Chinese prompt + `\boxed{}` answer (isomorphic to the platform eval protocol) |
| Batching | 64 images/batch, 16 concurrent, ~0.85 samples/s |
| Reliability | 54 h unattended, **all 35 NPU ERR99999 crashes self-healed** (3-layer self-healing: main loop + 30-min keepalive + external supervisor), 75% effective throughput |

Category distribution: spatial 44,231 (39.1%) / numeric 28,731 (25.4%) / counting 27,203 (24.0%) / perception 12,376 (10.9%) / physics 618 (0.5%). **The natural scarcity of physics** is the root cause of the later weakness profile (dimension-choice logic in 4.3).

### 4.2 SFT Data Curation (train_v4)

| Item | train_v4 |
| --- | --- |
| Size | 24,251 samples (17,924 curated from the 113K distillate + 6,327 round-2 targeted top-up) |
| Composition | Domain-specific reasoning 43.6% / code & spatiotemporal 34.3% / perception 14.8% / abstract shapes 7.25% |
| Quality metrics | Token length p50=325 / p90=1156 / max=2041 (all within max_len=2048); drop rate only **0.524%** (127 oversized images filtered, too_long=0) |
| Preliminary score | **41.8 pts** (official preliminary eval) |

### 4.3 RL Training Data (rlx30 pool)

| Item | Value |
| --- | --- |
| Pool | `rl3_data_abs.jsonl`: 1,709 **[RAVEN](https://github.com/WellyZhang/RAVEN) 3×3 abstract-reasoning** items (single-dimension, high purity) |
| Task form | Infer the 9th cell from the first 8 of a 3×3 grid, 8-way choice, answer = index 1–8 → `\boxed{5}` |
| Dimension choice | Six-dimension baseline profile (v4 base, 8 samples per prompt): association 0.212, the 2nd-weakest, with 100% reward-signal coverage; physics 0.079 was weaker, but synthetic data mismatched the eval distribution with no transfer measured (ph1 experiment, 18.8 pts) — hence the choice of the abstract question type corresponding to association |
| Quality audit | boxed extraction 100% / reward coverage 100% (answers all exactly-matchable indices) / zero duplicate images / absolute-path read-back verified |

The three-level data funnel: **113,159 (full distillate) → 24,251 (17,924 SFT-curated + 6,327 round-2 top-up, hard filter drops only 0.524%) → 1,709 (RL targeted concentration)**; SFT breadth-first, RL depth-first.

## 5. Base Model & SFT Training

- **Base**: Qwen3.5-0.8B multimodal (0.8B params, with vision encoder), LoRA fine-tune (r16 / α32), 2 epochs, seed 42, all-linear, NPU training.
- **v4 = the full train_v4 recipe**: confirmed through multiple rounds of variable-isolation experiments (once config-side gains were exhausted, the lever was on the data side); preliminary **41.8 pts** advanced, making it the sole compliant starting point for the finals.
- **v4's baseline under the finals eval protocol is 18.2 pts** (the two rounds use different eval sets, scores not directly comparable; every in-finals gain is measured against 18.2 on the same protocol).

## 6. Reinforcement Learning (GRPO)

### 6.1 Motivation & Framework

The finals require RL training that starts from the preliminary model as its sole starting point. We use **TRL GRPOTrainer (0.29.1, use_vllm=False)**, with a self-built training script `grpo_apply_v14.py`. Rationale: given the 0.8B model + NPU memory constraints, PPO's separate value network is not worth it; GRPO is critic-free, its within-group normalization naturally fits verifiable rewards, and it is simple to implement and numerically stable.

### 6.2 Reward Design

```
R = acc + 0.5 × fmt   (structurally equivalent to a 3-tier discrete reward)
  ├─ \boxed{} extracted AND equals reference after normalization → 1.0
  ├─ \boxed{} extracted but wrong                                → 0.05 (format floor, keeps a format gradient)
  └─ no \boxed{} extractable                                     → 0.0
```

| Design point | Implementation |
| --- | --- |
| **Protocol alignment with the platform** (core) | Extractor = `rfind("\boxed{")` + bracket-depth counting, verbatim identical to the official `eval_val.py`; normalization = strip whitespace/punctuation, fold full-width→half-width + lowercase, exactly the platform scorer's protocol ⇒ **the training reward = the eval objective, no room for reward hacking** |
| Composite-reward extension hooks | `--part_credit` (semantic partial credit), `--len_penalty` (proportional over-length penalty) — off by default, keeping strict identity with the platform protocol |
| Answer-type awareness | bool / choice / num / text four-way classification + "reference-answer self-consistency" audit (any type <100% ⇒ that type trains for nothing) |
| Zero-gradient guard | Training monitors `frac_reward_zero_std` (share of steps where all in-group rewards are equal ⇒ zero gradient) |

### 6.3 rlx30 Training Recipe (All Hyperparameters)

| Design item | Value | Rationale |
| --- | --- | --- |
| Algorithm / framework | GRPO (TRL 0.29.1, self-built `grpo_apply_v14.py`) | critic-free, memory-friendly |
| Group size K | 8 (8 generations per prompt) | K16 doubled effective learning yet lost 3.25 pts (see 6.6); K8 is the sweet spot |
| Steps | **30** (small dose) | measured peak of the dose axis (see 6.4) |
| Learning rate | 1e-6 constant | limits weight drift |
| KL coefficient β | 0.2 | anchors to the preliminary start, prevents catastrophic forgetting |
| Effective batch | 4 × 2 accum = 8 prompts/step, 240 prompts sampled total | 30 steps consume only 14% of the pool — no shortcut overfitting |
| max_completion_length | 256 | training completions average 141 tokens; the cap never binds |
| loss_type | dapo (TRL default) | token-level normalization, stable on the long tail |
| Starting point | Preliminary v4 weights byte-for-byte (`guard_base_v4.sh` gate verifies md5) | hard compliance rule: no intermediate SFT, no other starting point |
| Cost | **7.5 min** end-to-end (training 212 s = 7.1 s/step + export & packing ~4 min) | ablation iterations are extremely cheap |

### 6.4 Dose-Axis Experiment (Non-Monotonic; 30 Steps Is the Peak)

| Steps | 10 | 20 | **30** | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score | 19.2 | 15.2 | **19.6–20.2** | 17.8 | 16.8 |

The RL gain has an optimal dose window; beyond 30 steps the score falls monotonically with dose — larger doses cause catastrophic forgetting (mechanism in 6.7).

### 6.5 Single-Variable Ablations (the ZIP-CRC Proof Method)

Methodological innovation: four submission packages share one identical set of weights (`model.safetensors` byte-identical; per-member zip CRC audit: the decoding/gen-length variants differ only in `generation_config.json`, the preprocessing variant additionally changes two lines in `processor_config.json`/`preprocessor_config.json`) — **causal conclusions without any retraining**.

| Axis | Sole variable changed | Score | Δ | Conclusion |
| --- | --- | ---: | ---: | --- |
| Baseline | platform-default sampling (same-package 4-run μ) | 19.85 | — | the temperature curve peaks exactly at the platform default |
| Decoding | greedy (do_sample=false) | 16.8 | **−3.05** | sampling is a load-bearing pillar |
| Decoding | temperature 1.15 / top_p 0.95 | 16.6 | **−3.25** | the default is the peak |
| Gen length | +max_new_tokens 2048 | 19.8 | −0.05 | zero effect (reverse evidence that the weights are byte-identical) |
| Preproc | longest_edge 16,777,216 → 1,638,400 | 16.4 | **−3.45** | 56.7% of eval images lose ~90% of pixels — detail destroyed |

⇒ **Sampling buys ~3 pts for free; keep every inference-side parameter at the platform default.**

### 6.6 Data-Pool Axis & Group-Size K Axis

| Ablation | Config | Score | Conclusion |
| --- | --- | ---: | --- |
| abstract single-dim (rlx30) | RAVEN 1,709 | **20.2** | best distribution match |
| six-dim balanced (p01v1) | 1,336 items, six dims | 17.8 | balancing rejected |
| six-dim + augmented (p01v2full) | 2,175 items | 16.8 | the more complete, the worse |
| physics single-dim (ph1) | 1,000 | 18.8 | within the noise band; no transfer from synthetic data |
| counting single-dim (h7) | 1,709 | 16.4 | counting line rejected |
| K8→K16 (h9k16) | rest identical to rlx30 | 16.6 | −3.25; amplified learning efficiency loses points |

⇒ **Distribution match** between the pool and the hidden official eval set is the deciding factor (not format, not coverage).

### 6.7 Core Finding: The 0.8B Catastrophic-Forgetting Law

| Package | In-pool avg reward (learning) | Score |
| --- | ---: | ---: |
| rlx30 (abstract) | **0.088** (barely learned) | **20.2** |
| ph1 (physics) | 0.843 | 18.8 |
| h7 (counting) | 0.644 | 16.4 |
| h9k16 (K16) | 2× effective learning | 16.6 |

Four independent data points, one consistent trend: **in-pool learning anti-correlates strongly with the platform score.** The K16 ablation supplies the mechanism: in-pool reward peak 0.287→0.581 (roughly doubled), zero-gradient steps 51%→27%, yet the score drops 3.25 — a 0.8B model learning RL shortcuts in the pool crowds out its already-limited capacity, at the cost of forgetting other dimensions.

⇒ **The optimal 0.8B RL policy = minimal-dose anchoring** (30 steps + β 0.2 + lr 1e-6): RL as fine calibration, not large policy updates. rlx30's 0.088 in-pool learning is the direct product of exactly this set of constraints.

### 6.8 Platform Noise & Statistical Decisions

- Same package, 4 submissions: 19.6 / 20.2 / 19.6 / 20.0 ⇒ the platform eval carries **σ_judge ≈ 0.29** noise (same-weights reading threshold ±0.29, cross-artifact 1.3–1.6).
- Budget decision: 13 historical readings μ=17.54, σ=1.57 ⇒ P(new recipe > 20.2) ≈ 1.1% (400k Monte-Carlo estimate; the normal approximation would give ≈4.5%) vs re-submitting rlx30 ≈ 11.3%/shot ⇒ **decision: re-submit the best artifact**.
- Every conclusion follows the three-state labeling discipline: measured / historical / unverified.

### 6.9 Contribution Analysis (SFT → RL Gain Decomposition)

| Version | Score | Gain |
| --- | ---: | --- |
| v4-SFT (preliminary; finals-protocol baseline) | 18.2 | distillation foundation |
| **rlx30 = v4 + GRPO 30 steps** | **20.2** | **RL +2.0** |
| + decoding kept at default sampling | (sampling vs greedy differs by 3.0) | inference-side free gain |

Every layer's gain has a single-variable evidence chain: the sole variable behind RL +2.0 is GRPO 30 steps (guaranteed by the start-point md5 gate); the inference-side +3.0 is proven by the same-weights ZIP-CRC ablation.

## 7. Submitted Model

- **Submitted version**: rlx30 = v4 + GRPO 30 steps (abstract pool), the complete exported model directory, packed as `qianwen_submit_rlx30_final.zip`.
- **13 files, flat**: model.safetensors (bf16, 1,706,030,528 B) + config.json + configuration.json + generation_config.json + preprocessor_config.json + processor_config.json + video_preprocessor_config.json + the full tokenizer set (tokenizer.json / tokenizer_config.json / vocab.json / merges.txt) + chat_template.jinja + README.md. No nested directories; CRC verified.
- **Delivery verification**: zip md5 `37f81f331ed2ca75fc87d1045ab1a1b3`, weights md5 `680b8af9af63699e2da5798029095b81`; server ↔ backup disk ↔ local, three-way byte-for-byte identical.
- **Platform verification**: four submission readings 19.6 / 20.2 / 19.6 / 20.0 (μ=19.85, σ=0.29); historical best 20.2 counted.

## 8. Summary

- With the two-stage route "teacher-distillation SFT foundation + small-dose GRPO fine calibration", the 0.8B model was lifted from the finals-protocol baseline 18.2 to **20.2** (RL +2.0); both readings already include the platform-default sampling gain — switching to greedy would lose ≈3 pts (ZIP-CRC ablation).
- The systematic experiments distilled three model-agnostic methodological assets: **verbatim reward-eval protocol alignment** (the training reward is the eval objective, eliminating reward hacking), **the single-variable ZIP-CRC proof method** (inference-side causal ablation without retraining), and **statistical budget decisions** (platform-noise quantification + Monte-Carlo re-submission modeling).
- The core finding — **the 0.8B catastrophic-forgetting law** (in-pool learning anti-correlates with the platform score) — directly shaped rlx30's minimal-dose recipe and points toward curriculum-style RL and reward engineering at larger scales as the next step.
- The deliverable contains the complete model directory and this report; the train / export / pack / verify chain is fully scripted and reproducible.

## 9. Appendix: Deliverables List

| File | Description |
| --- | --- |
| `qianwen_submit_rlx30_final.zip` | Finals submission package (13 files flat, md5 `37f81f33…`) |
| `grpo_apply_v14.py` | GRPO training script (contains the reward: extractor/normalization verbatim-identical to the official eval) |
| `guard_base_v4.sh` | Starting-point compliance gate (pre-training md5 check = byte-for-byte the preliminary submission) |
| `export_v14.py` | Export script (GRPO ckpt → submission dir, dtype-label fix fp32→bf16) |
| `make_variant.py` | Submission-package variant generation & ZIP packing (`--tag` batch tags prevent name collisions) |
| `verify_zip.py` | Read-only submission verification (13 files / flat / CRC / total uncompressed size) |
| `rl3_data_abs.jsonl` | RL training pool: 1,709 RAVEN 3×3 items (100% reward coverage) |
| `train_v4.jsonl` | SFT data: 24,251 samples (train_v4 recipe) |
| Distillation pipeline scripts | Teacher batch inference & cleaning (113,164 images → 113,159 samples) |
| 14 submitted packages, official registry | 17 submission readings (rlx30 re-submitted 4×; the 13 non-rlx30 readings form the μ=17.54 statistics in §6.8) with single-variable controls (the source of all data in §6) |
