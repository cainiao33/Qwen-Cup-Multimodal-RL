# Data Engineering: Candidate Pool → Distillation Synthesis → Cleaning & Mixing → RL Targeted Pool → Reward Alignment

[简体中文](data-engineering.md) | **English**

> Competition rule: **the organizer provides no training data — data must be self-built, making data strategy one of the core competitive dimensions.** This document fully describes the data sources, synthesis method, cleaning & verification workflow, mixing decisions, targeted RL data concentration, and how the post-training reward was aligned with the official eval protocol. All numbers are consistent with the [finals technical report](finals-technical-report_en.md).

---

## 0. Overview: A Three-Level Data Funnel

```
113,164 candidate images    113,159 distilled samples    24,251 SFT curated      1,709 RL targeted pool
(4 public datasets, round 1) ──►  (Qwen3.6-27B teacher)  ──►  (clean+verify+mix)  ──►  (RAVEN single-dim)
                                99.996% completion          0.524% drop rate         100% reward coverage
                                                            ▲                        ▲
                                                     SFT breadth-first         RL depth-first
                                                     (preliminary 41.8)        (finals 20.2)
```

Each compression level serves one clear goal: level-1 synthesis solves **getting data from nothing**; level-2 curation solves **quality and mixing ratio**; level-3 concentration solves **RL signal purity**. Note: 24,251 = 17,924 curated in round 1 + 6,327 round-2 targeted top-up (round-2 images come from 4 other public datasets, see §1.1); the RL pool is drawn independently from RAVEN.

## 1. Candidate Image Pool (Data Sources)

The organizer provides no training data ⇒ all training data was self-built. The candidate pool came from 4 public multimodal datasets, **113,164 images** in total:

| Dataset | Nature | Targeted dimension |
| --- | --- | --- |
| CLEVR-Math | Synthetic scene geometry, counting & numeric reasoning | Counting / numeric computation |
| ChartQA | Chart question answering | Numeric (tables / bills) |
| ScienceQA | Science-knowledge multimodal QA | Physics commonsense |
| VSR | Visual spatial-relation judgment | Spatial reasoning / fine-grained perception |

Pool-selection principle: aim at the six preliminary-round dimensions (numeric, counting, spatial reasoning, physics commonsense, fine-grained visual perception, image-text association), using public datasets as the base and the teacher model for synthetic annotation — avoiding the cost and scale bottlenecks of human labeling. All four bases are public academic benchmarks: ChartQA (ACL 2021), ScienceQA (NeurIPS 2022), VSR (TACL 2022), CLEVR-Math (a synthetic math-reasoning set derived from CLEVR).

### 1.1 Additional Sources for the Round-2 Targeted Top-Up (4 More Public Datasets)

train_v4 did not come entirely from the round-1 candidate pool: beyond the 17,924 v2 legacy curated samples, round-2 added ~6,327 distilled samples targeting four directions (verified by checking teacher answers against reference answers before merging, see §3.2), whose image sources are **four other public datasets**:

| Dataset | Nature | Targeted direction (six-dim pool) |
| --- | --- | --- |
| GQA (CVPR 2019) | Scene-graph question answering | spatial |
| PlotQA (WACV 2020) | Chart numeric question answering | numeric |
| FSC-147 (NeurIPS 2021) | Open-world object counting | counting |
| PhyX | Physics-commonsense image-text | physics |

> Evidence: the `POOL_MAP` in [sft/scripts/merge_train_v4.py](../sft/scripts/merge_train_v4.py) assigns round-2 samples to the four pools above by their `source` field (a provenance layer the finals technical report did not expand on). Two rounds total **8 public datasets, no private data sources anywhere**.

## 2. Teacher Distillation Synthesis (How the Data Was Synthesized)

| Item | Value |
| --- | --- |
| Teacher | Qwen3.6-27B, local vLLM inference (bf16, TP=2) |
| Batching | 64 images/batch, 16 concurrent, ~0.85 samples/s |
| Output | **113,159** LLaVA-style samples (99.996% completion) |
| Format | Single-turn dialogue: `<image>` + Chinese prompt + `\boxed{}` answer — isomorphic to the official eval extraction format (the official eval requires the "final answer inside \boxed{}", otherwise it cannot be parsed) |
| Reliability | 54 h unattended, **all 35 NPU ERR99999 crashes self-healed** (three-layer care: main-loop retry + 30-min keepalive + external supervisor), 75% effective throughput |

Category distribution of the synthesis output (labeled by eval dimension):

| Dimension | Samples | Share |
| --- | ---: | ---: |
| spatial | 44,231 | 39.1% |
| numeric | 28,731 | 25.4% |
| counting | 27,203 | 24.0% |
| perception | 12,376 | 10.9% |
| physics | 618 | **0.5%** |

**The natural scarcity of physics is the key finding of this step**: physics-commonsense material is extremely rare in public data pools, which planted the seed of the later "weakness profile → RL dimension choice" decision chain (see §5).

## 3. Cleaning & Verification (How It Was Cleaned)

Level two of the funnel: 113,159 → 24,251, **total drop rate only 0.524%**. Cleaning was not aggressive pruning but three things: "verify + filter + merge":

### 3.1 Hard Filters

- **Oversized-image filter**: 127 oversized images filtered; **too_long = 0** (no sample dropped for length).
- **Token-length audit**: p50 = 325 / p90 = 1156 / max = 2041, all within max_len = 2048 — no sample was sacrificed to truncation, and the length distribution leaves ample room for inference-time expansion.

### 3.2 Round-2 Verified Merge

train_v4 is a **verified merge of two batches**:

```
train_v4 (24,251) = train_v2 legacy (17,924) + round-2 freshly distilled & verified (~6,327)
```

Round 2 topped up distillation for spatial / physics / numeric / counting (image sources: GQA / PlotQA / PhyX / FSC-147, see §1.1); the new output was verified first (teacher answer checked equal to reference answer) before merging, so distillation noise never entered training directly. Merge logic in [sft/scripts/merge_train_v4.py](../sft/scripts/merge_train_v4.py) (image paths accept both relative and absolute forms; full read-back verification before training).

### 3.3 Final Mixing (train_v4)

| Category | Share | Corresponding finals dimension (by task nature) |
| --- | ---: | --- |
| Domain-specific reasoning | 43.6% | Domain-specific reasoning (chemistry / math) |
| Code & spatiotemporal reasoning | 34.3% | Code interpretation & spatiotemporal reasoning |
| perception | 14.8% | Fine-grained visual perception / OCR text extraction |
| Abstract shapes | 7.25% | Abstract pattern rules (3×3 IQ-test matrices) |

The top two categories exactly cover the first two official finals dimensions; abstract shapes take only 7.25% but preserved a foundation in that direction for the RL stage (see §5 — which is exactly where the final RL gain came from).

Note: the percentages in this table categorize the same training data **by the official brief's four task directions** (a task-type grouping; the eval taxonomy is the six dimensions), while §2 labels by the six dimensions — two statistical lenses on the same data; both distillation rounds' candidate pool stayed within the public datasets of §1, and round 2 introduced no private sources either.

### 3.4 Mixing-Experiment Methodology

The v3 → v4 progression is a **variable-isolation experiment**: identical configs (2 epochs / LoRA r16 α32 / lr 1e-4 / eff_bs 16), data scale only increased (v2 → v4). Conclusion: after config-side gains (+3.4) were exhausted, the lever shifted to the data side (+0.4). — Data-mixing gains must be proven with a "data-only" control, not conflated with config tuning.

## 4. RL Targeted Concentration (1,709 Items)

Funnel level three: not coverage, but **distribution match + reward-signal purity**.

### 4.1 Pool Construction

`rl3_data_abs.jsonl`: **1,709 RAVEN 3×3 abstract-reasoning matrices** (RAVEN, Zhang et al., CVPR 2019, a public abstract-reasoning dataset) — infer the 9th cell from the first 8, 8-way choice, answer is an index 1–8 (as `\boxed{5}`).

Quality audit, all four checks passed:

| Audit item | Result |
| --- | --- |
| boxed extraction rate | 100% |
| Reward-signal coverage (answers exactly matchable) | 100% |
| Duplicate images | 0 |
| Image-path read-back | full absolute-path verification passed |

### 4.2 Why the Single Abstract Dimension (Dimension-Choice Decision Chain)

1. **Weakness profile as the base**: the six-dimension profile of the v4 base, 8 samples per prompt, showed association at 0.212 as the second-weakest dimension, with 100% reward-signal coverage for that question type;
2. **Physics was weaker but unusable**: 0.079, even weaker — but synthetic physics data mismatched the eval distribution, with no transfer measured (ph1 experiment, 18.8, within the noise band);
3. **Official-dimension confirmation**: the finals brief lists "abstract pattern rules: discovering composite rules of matrix figures and logical reasoning, e.g. classic 3×3 IQ tests" as one of the four core task directions (the official brief's task-type grouping; the eval taxonomy is the six dimensions) — the pool choice aligns directly with the official direction;
4. **Ablation verdict**: abstract single-dim 20.2 > physics single-dim 18.8 > six-dim balanced 17.8 > six-dim augmented 16.8 > counting single-dim 16.4. **Distribution match between the pool and the hidden eval set is the deciding factor, not coverage.**

## 5. Post-Training Alignment (Reward = Eval Protocol)

The eval extraction rules are fixed by the official code (`rfind` locates the `\boxed{` / `boxed{` prefix, bracket-depth counting supports arbitrary nesting). This solution's reward function is **verbatim identical** to the official extractor:

```
R = acc + 0.5 × fmt  (structurally equivalent to a 3-tier discrete reward)
  ├─ \boxed{} extracted AND equals reference after normalization → 1.0
  ├─ \boxed{} extracted but wrong                          → 0.05 (format floor, keeps a format gradient)
  └─ no \boxed{} extractable                               → 0.0
```

| Alignment item | Implementation |
| --- | --- |
| Answer extraction | Same algorithm as the official `extract_boxed_answer`: `rfind` + bracket-depth counting, 100% support for arbitrary nesting |
| Answer normalization | Strip whitespace / punctuation / full-half-width folding + lowercase — exactly the platform scorer's protocol |
| Conclusion | **The training reward is the eval objective — no room for reward hacking.** The only way to raise the reward is to answer correctly. |

Companion robustness designs:

- **Answer-type awareness**: bool / choice / num / text four-way classification + "reference-answer self-consistency" audit (any type < 100% ⇒ that type trains for nothing);
- **Zero-gradient guard**: monitors `frac_reward_zero_std` (share of steps where all in-group rewards are equal ⇒ zero gradient), preventing accumulation of wasted steps;
- **Format floor 0.05**: wrong-but-well-formatted answers keep a tiny gradient so the format-learning signal never dies.

## 6. Data-to-Score Mapping (Evidence at Every Funnel Level)

| Funnel level | Data | Corresponding model / score |
| --- | --- | --- |
| Full distillate | 113,159 | synthesis base (not trained on directly) |
| SFT curated | 24,251 (train_v4) | v4: preliminary **41.8**, finals-protocol baseline 18.2 |
| RL targeted pool | 1,709 (RAVEN abstract) | rlx30: finals **20.2** (same-protocol +2.0) |

Every data decision at every level is backed by a single-variable evidence chain; full experiment data in the [finals technical report](finals-technical-report_en.md) §4–§6.
