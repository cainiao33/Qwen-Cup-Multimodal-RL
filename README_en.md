# Qwen Cup Multimodal Reasoning Competition: Teacher-Distillation SFT + Small-Dose GRPO RL

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Switch_Language-简体中文-red?style=for-the-badge" alt="简体中文"></a>
  <a href="README_en.md"><img src="https://img.shields.io/badge/Switch_Language-English-blue?style=for-the-badge" alt="English"></a>
</p>

> **Overview**: a two-stage post-training pipeline on Qwen3.5-0.8B (multimodal) — teacher-distillation SFT for breadth, then small-dose GRPO reinforcement learning for fine calibration — covering six reasoning dimensions: numeric, counting, spatial, physics, perception, and association. The transferable output is a set of experimental methodologies: verbatim reward-eval alignment, same-weights ZIP-CRC ablation, statistical submission-budget decision-making, and a systematic characterization of the RL dose-forgetting trade-off on 0.8B-scale models.
>
> **Results**: preliminary round **41.8 pts** (500 problems, track best 43.6, **20/337** at the cutoff) → finals online eval **20.2 pts** (same-protocol SFT baseline 18.2, RL gain **+2.0**), **final standing 9/20**; online defense **3rd place** (7th to present among the top 10, with 2 teams withdrawing), **¥20,000 prize**.
>
> **How to run**: this repo is positioned as a **complete open-sourcing of the solution** — training/export script snapshots, real training logs, and the full technical report are all included. Training ran on Huawei Atlas 900 NPU + torch_npu (scripts are snapshots of that environment); the eval set is not public, so platform scores cannot be reproduced locally. Model weights are hosted on [Hugging Face](https://huggingface.co/huangxiu/Qwen3.5-0.8B-qwencup) (links in [Reproduction Notes](#-reproduction-notes)).

---

## The Competition

Qwen Cup Multimodal Reasoning (initiated by Alibaba's Qwen business unit, co-hosted with Pazhou Lab; [official competition page](https://www.aicompetition-pz.com/topic_detail/24)): given interleaved image-text inputs, the model must answer in the `\boxed{answer}` format. Evaluation is a hidden 500-problem set spanning six dimensions (association / counting / numeric / perception / physics / spatial); platform score = correct / 500 × 100 (0.2 pts per problem), best historical submission counts.

Format: preliminary round (SFT stage, Jul 10 – Aug 30 2026, up to 2 submissions/day, 337 teams, top 20 advance) → finals (RL starting from the preliminary model, Sep 5 – 25, up to 5 submissions/day) + online defense (Oct 12, top 10). **The organizer provides no training data — data must be self-built**, making data strategy a core competitive axis. Final scoring = model performance 65% + technical proposal 25% + defense 10%.

Hard rules (the starting point for every engineering decision here):

1. RL must start from the preliminary-round submitted model; no other starting point allowed;
2. No architecture changes, no additional pre-trained weights;
3. The eval set is not public; scoring is done by the organizer, with limited submissions per day.

## Overall Approach

```
Stage 1 SFT (preliminary, breadth-first)       Stage 2 RL (finals, depth-first)
Qwen3.6-27B teacher (vLLM, TP=2, bf16)         Preliminary v4 weights (md5 gate)
        │ distill 113,164 imgs → 113,159 samples        │ GRPO, 30 steps, small dose
        ▼                                               ▼
Curated train_v4 (24,251)  ──────────────►   rlx30 (single-dim RAVEN pool, 1,709)
2 epochs / seed 42 / all-linear               lr 1e-6 / β 0.2 / K 8 / seed 42
Preliminary 41.8                              Finals 20.2 (SFT baseline 18.2 → +2.0)
```

Three design principles: **SFT for breadth, RL for depth** (a 0.8B model has little spare capacity — RL does fine calibration only, avoiding catastrophic forgetting); **protocol alignment** (the training reward matches the official eval script verbatim — zero room for reward hacking); **budget rationality** (each submission is a noisy measurement; allocate them statistically).

## Stage 1: Teacher-Distillation SFT (Preliminary 41.8)

### Distillation pipeline (113K samples)

| Item | Value |
| --- | --- |
| Teacher | Qwen3.6-27B, local vLLM inference (bf16, TP=2) |
| Image pool | Round 1: CLEVR-Math / ChartQA / ScienceQA / VSR (113,164 images); Round 2 top-up: GQA / PlotQA / PhyX / FSC-147 — 8 public datasets in total, see [data engineering](docs/data-engineering.md) |
| Output | **113,159** LLaVA-style samples (99.996% completion); single-turn `<image>` + Chinese prompt + `\boxed{}` answer, isomorphic to the platform eval format |
| Throughput | 64 images/batch, 16 concurrent, ~0.85 samples/s |
| Reliability | 54 h unattended, **all 35 NPU ERR99999 crashes self-healed** (3-layer care: main loop + 30-min keepalive + external supervisor), 75% effective throughput |

Category distribution: spatial 44,231 (39.1%) / numeric 28,731 (25.4%) / counting 27,203 (24.0%) / perception 12,376 (10.9%) / physics 618 (0.5%) — physics is naturally scarce, the root cause of the later weakness profile.

### Curation & training (train_v4)

- **First compression of the 3-level data funnel**: 113,159 (full distillate) → **24,251** (SFT curation, drop rate 0.524%); token length p50=325 / p90=1156 / max=2041, all within max_len=2048.
- **Recipe**: LoRA r16 / α32, 2 epochs, lr 1e-4, effective batch 16, max_len 2048, all-linear, seed 42, single NPU.
- **Variable isolation**: v4 is identical to v3 except data size — after config-side gains (+3.4) were exhausted, the data side delivered +0.4 more.
- **Training log** (see [sft/logs/](sft/logs/)): 3,032 steps in one uninterrupted run, loss 6.07 → 2.03.

### Preliminary result

**41.8 pts** (500 problems, qwen-plus judge), track best 43.6, ranked 20th — advanced to the finals.

## Stage 2: Small-Dose GRPO RL (Finals 20.2, 9th online)

### Reward design: verbatim alignment with the eval (core)

```
R = acc + 0.5 × fmt   (effectively a 3-tier discrete reward)
  ├─ \boxed{} extracted AND matches reference after normalization → 1.0
  ├─ \boxed{} extracted but wrong                                  → 0.05 (format floor)
  └─ no \boxed{} extracted                                         → 0.0
```

The answer extractor (`rfind("\boxed{")` + bracket-depth counting) matches the official eval script **verbatim**; normalization (strip whitespace/punctuation, full-half-width folding, lowercase) matches the platform scorer exactly ⇒ the training reward *is* the eval objective — no room for reward hacking. Plus zero-gradient guards (monitoring the share of steps with zero within-group reward std) and per-answer-type consistency audits.

### rlx30 recipe

| Design choice | Value | Rationale |
| --- | --- | --- |
| Framework | TRL 0.29.1 GRPOTrainer + vLLM online sampling | critic-free, memory-friendly, stable |
| Group size K | 8 | K16 doubled effective learning yet lost 3.25 pts — K8 is the sweet spot |
| Steps | **30 (small dose)** | measured peak of the dose axis (below) |
| LR / KL β | 1e-6 constant / 0.2 | limit weight drift, anchor to the preliminary start |
| Effective batch | 8 prompts/step, 240 prompts total | 30 steps consume only 14% of the pool — no shortcut overfitting |
| Data pool | RAVEN 3×3 abstract-reasoning matrices, 1,709 items (single-dimension, high purity) | association (0.212) was the 2nd-weakest dim with 100% reward-signal coverage — and abstract-pattern matrices (3×3 IQ-test style) are one of the four official finals dimensions |
| Compliance gate | starting-weights md5 verified byte-for-byte in a pre-training script | hard rule: no intermediate SFT, no other starting point |
| Cost | 7.5 min end-to-end (212 s = 7.1 s/step) | ablation iterations are extremely cheap |

### Finals result

- v4's baseline under the finals eval protocol is **18.2 pts** (the two rounds use different eval sets; scores are not directly comparable).
- **rlx30 = v4 + GRPO 30 steps: 20.2 pts** (four submissions: 19.6 / 20.2 / 19.6 / 20.0, μ=19.85, σ=0.29), **finals final standing 9/20**, same-protocol RL gain +2.0.
- Defense: top 10 advanced to the online defense, 2 teams withdrew, we presented 7th — **3rd place in the defense, ¥20,000 prize**.

## Findings

### 1. The RL dose axis is non-monotonic; 30 steps is the peak

| Steps | 10 | 20 | **30** | 50 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score | 19.2 | 15.2 | **19.6–20.2** | 17.8 | 16.8 |

Beyond 30 steps the score falls monotonically with dose — larger doses cause catastrophic forgetting (mechanism in Finding 3).

> Anecdote: the 30-step recipe itself was born of an accident — a 40-step run crashed with a full disk at step 35, and `checkpoint-30` was salvaged (bit-equivalent to a standalone 30-step run under the same seed/data order). The dose-axis ablation later confirmed 30 as the peak; see [grpo/README.md](grpo/README.md).

### 2. Same-weights ZIP-CRC ablation: causal conclusions without retraining

Four submission packages share one identical set of weights (`model.safetensors` byte-identical; per-member zip CRC audit confirms only `generation_config.json` differs) — **causal inference-side conclusions with zero retraining**:

| Axis | Sole variable changed | Score | Δ | Conclusion |
| --- | --- | ---: | ---: | --- |
| Baseline | platform-default sampling (same-package 4-run μ) | 19.85 | — | the temperature curve peaks exactly at the platform default |
| Decoding | greedy (do_sample=false) | 16.8 | **−3.0** | sampling is a load-bearing pillar |
| Decoding | temperature 1.15 / top_p 0.95 | 16.6 | **−3.05** | the default is the peak |
| Gen length | +max_new_tokens 2048 | 19.8 | −0.05 | zero effect (reverse evidence the weights are identical) |
| Preproc | longest_edge 16,777,216 → 1,638,400 | 16.4 | **−3.2** | 56.7% of eval images lose ~90% of pixels — detail destroyed |

⇒ Sampling buys ~3 pts for free; keep every inference-side parameter at the platform default.

### 3. The 0.8B catastrophic-forgetting law: in-pool learning anti-correlates with the platform score

| Package | In-pool avg reward (learning) | Score |
| --- | ---: | ---: |
| rlx30 (abstract) | **0.088** (barely learned) | **20.2** |
| ph1 (physics) | 0.843 | 18.8 |
| h7 (counting) | 0.644 | 16.4 |
| h9k16 (K16) | 3× effective learning | 16.6 |

Four independent points on one line: the more learned in-pool, the lower the platform score. The K16 ablation supplies the mechanism: in-pool reward peak 0.287→0.581 (3× learning efficiency), zero-gradient steps 51%→27%, yet the score drops 3.25 — a 0.8B model learning RL shortcuts in the pool crowds out its limited capacity at the cost of forgetting other dimensions.

The data-pool axis agrees: abstract single-dim (RAVEN 1,709) 20.2 > physics 18.8 > six-dim balanced 17.8 > six-dim augmented 16.8 > counting 16.4 — **distribution match with the hidden eval set is what matters, not coverage**.

⇒ The optimal small-model RL policy = minimal-dose anchoring: RL as fine calibration, not large policy updates.

### 4. Platform noise & statistical budget decisions

- Same package, 4 submissions: σ_judge ≈ 0.29; cross-artifact threshold 1.3–1.6.
- 13 historical readings μ=17.54, σ=1.57 ⇒ P(new recipe > 20.2) ≈ 1.1% vs re-submitting the best artifact ≈ 11.3%/shot (400k Monte-Carlo trials) ⇒ **decision: re-submit the best artifact; spend quota only where statistics favor it**.

### Gain decomposition (every layer has a single-variable evidence chain)

| Version | Score | Gain | Evidence |
| --- | ---: | --- | --- |
| v4-SFT (finals-protocol baseline) | 18.2 | distillation foundation | preliminary 41.8 |
| **rlx30 = v4 + GRPO 30 steps** | **20.2** | **RL +2.0** | start-point md5 gate guarantees the sole variable |
| + platform-default sampling | (greedy costs 3.0) | sampling +3.0 | same-weights ZIP-CRC ablation |

## Repository Layout

```
Qwen-Cup-Multimodal-RL/
├── README.md / README_en.md
├── docs/
│   ├── finals-technical-report.md   # full finals technical report (authoritative source of all numbers)
│   ├── data-engineering.md          # data engineering: sources/synthesis/cleaning/mixing/RL pool/reward-eval alignment (Chinese)
│   └── preliminary-sft-notes.md     # preliminary v4 script-package notes (provenance & repro commands)
├── sft/                             # Stage 1: teacher-distillation SFT (preliminary v4)
│   ├── scripts/                     # 5 scripts: train/watchdog/merge/export (server snapshot, sanitized)
│   └── logs/                        # real training & export logs (3,032 steps, loss 6.07 → 2.03)
└── grpo/                            # Stage 2: GRPO RL (finals rlx30: train/export/pack scripts + full hyperparameter snapshot)
```

## 🔄 Reproduction Notes

This repo **does not offer one-click end-to-end reproduction**; scope and limits:

- **Eval side**: the eval set (6-dim, 500 problems) is not public — platform scores can only come from the organizer; every platform score in this repo is an official reading.
- **Training side**: training ran on Atlas 900 NPU + CANN + torch_npu. [sft/scripts/](sft/scripts/) holds sanitized snapshots from that environment (only internal paths and the interpreter path replaced with placeholders; logic untouched), with the matching real logs in [sft/logs/](sft/logs/). The GRPO side is now **complete**: the 3-step pipeline (`01_run_rlx30.sh` train → `02_export_rlx30.sh` export+pack → `03_pack_flat_zip.py` verify), the trainer `grpo_apply_v14.py` with the reward implementation verbatim-aligned with the official eval (both the 2026-09-15 and 09-18 revisions included), and all remaining originals — every md5 verified against the manifest. Workflow and known pitfalls in [grpo/README.md](grpo/README.md). Training logs and the 09-13 training-day revision remain server-only.
- **Data side**: how the 3-level data funnel (113K distillate → 24,251 SFT curation → 1,709 RL pool) was sourced, synthesized, cleaned, mixed, and reward-aligned is documented in **[docs/data-engineering.md](docs/data-engineering.md)** (Chinese); full experiment data in the [technical report](docs/finals-technical-report.md) §4–§6. The data files themselves are not distributed with the repo.
- **Weights**:

  | Model | Description | Download |
  | --- | --- | --- |
  | v4 (preliminary 41.8) | Qwen3.5-0.8B + merged LoRA export, bf16 | [weights `preliminary/`](https://huggingface.co/huangxiu/Qwen3.5-0.8B-qwencup/tree/main/preliminary) |
  | rlx30 (finals 20.2) | v4 + GRPO 30 steps, zip md5 `37f81f331ed2ca75fc87d1045ab1a1b3` | [weights `finals/`](https://huggingface.co/huangxiu/Qwen3.5-0.8B-qwencup/tree/main/finals) |

## Acknowledgements

- The competition organizers for the eval platform and compute
- [Qwen](https://github.com/QwenLM) model family (Qwen3.5-0.8B base & Qwen3.6-27B teacher)
- [TRL](https://github.com/huggingface/trl) · [vLLM](https://github.com/vllm-project/vllm) · [PEFT](https://github.com/huggingface/peft)
- CLEVR-Math / ChartQA / ScienceQA / VSR + GQA / PlotQA / PhyX / FSC-147 (distillation pool, 2 rounds) · RAVEN (RL pool)

## License

- **This repository (code, scripts, docs)**: [MIT License](LICENSE) — free to use, modify, and redistribute.
- **Model weights** (v4 / rlx30 on Hugging Face) are fine-tuned derivatives of Qwen3.5-0.8B and follow the [original Qwen model license](https://huggingface.co/Qwen).
