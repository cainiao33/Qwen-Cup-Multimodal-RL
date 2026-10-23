# Official Sample Dataset Walkthrough (training-data format examples)

[简体中文](official-sample-data.md) | **English**

> **Source**: the official example dataset released by the Qwen Cup organizers (training-data format examples, 1,000 records + 1,009 images; 132 MB archive, ≈142 MB unpacked). This page samples only **11 records** for format documentation (≈0.9 MB); the full package is not redistributed.

## 1. What this is

The organizers' official example of what one SFT training record looks like: each record is a **pre-rendered segment stream of the Qwen chat template** with loss masks already in place. The problem itself lives in the image; the text side carries only a short instruction (sometimes none). The reference answer is a full CoT ending in `\boxed{}`.

**Relation to this repo's training data**: our self-built `train_v4.jsonl` (24,251 records) uses the **SWIFT format** (`messages` / `images` keys; see [sft/scripts/merge_train_v4.py](../sft/scripts/merge_train_v4.py)), which the training framework renders into the same chat template — **equivalent content, different serialization**; both end answers with `\boxed{}`, matching the platform's extraction protocol.

## 2. Anatomy of one record

Top-level fields: `trace_id` (1–1000), `_source_file` (the sample was sliced from `vl_solve_problem_with_images.jsonl`), `message[]` (six segments, fixed order):

| Segment | Content | need_mask (loss mask) |
|---|---|---|
| `bos` | empty string (marker) | ✅ masked |
| `user_prefix` | `<\|im_start\|>user\n` | ✅ masked |
| `user_content` | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` vision placeholders + question text (**order not fixed**: text can also precede the placeholders) + `<\|im_end\|>`; `image_filenames` attaches here | ✅ masked |
| `assistant_prefix` | `<\|im_start\|>assistant\n` | ✅ masked |
| `assistant_content` | CoT solution + `\boxed{answer}` | ❌ **the only segment in the training loss** |
| `eos` | `<\|endoftext\|>` | ✅ masked |

One complete record (trace #1, alternating-sign series sum): [sample_record_trace1.json](assets/official-sample/sample_record_trace1.json).

## 3. Sample gallery (11 / 1000)

| Trace | Image | Problem / answer |
|---|---|---|
| #1 | ![](assets/official-sample/1.jpg) | Alternating-sign series sum; question "解答题目"; ans `\boxed{-2014m}` |
| #60 | ![](assets/official-sample/60.jpg) | Sequence limit; question "解答题目"; ans `\boxed{3}` |
| #784 | ![](assets/official-sample/784.jpg) | Limit via rationalization; question "解答题目"; ans `\boxed{1}` |
| #917 | ![](assets/official-sample/917.png) | Triangle-area model (geometry); the stem is on the text side and **precedes the vision placeholder**; ans `\boxed{6}` |
| #985 | ![](assets/official-sample/985.png) | Balance-scale word problem (full stem in text, image is the balance diagram; a real 3 KB image); ans `\boxed{5}` |
| #846 | ![](assets/official-sample/846.jpg) | Electrolysis-of-water + CO₂ chemistry, multi-part (6 `\boxed{}` spans; under the rfind extraction convention the last answer is `\text{铂}`) |
| #311 | ![](assets/official-sample/311.jpg) | Binary search tree insertion (CS multiple choice); ans `\boxed{C}` |
| #490 | ![](assets/official-sample/490.jpg) | Economics common-sense MC (general knowledge); ans `\boxed{B}` |
| #652 | ![](assets/official-sample/652.jpg) | Cashier daily-settlement practice (accounting, multi-select); ans `\boxed{①②③④⑤}` |
| #294 | ![](assets/official-sample/294_1.jpg) ![](assets/official-sample/294_2.jpg) | Grid construction task (**two-image record**; the set has 8 multi-image records, up to 3); ans is a coordinate list |
| #35 | ![](assets/official-sample/35.jpg) | Reciprocal-function (y=k/x) synthesis; **image-only, no text question**; ans `\boxed{-1<a<0}` |

## 4. Full-package statistics (1,000 records)

| Dimension | Data |
|---|---|
| Question text | "解答题目" 346 · "帮我分析一下图片内容" 241 · **image-only, no text 43** · "解答图中题目" 40 · "帮我分析一下图片" 23 · "这道题的答案是什么？" 20 · "这道题的正确答案是什么？" 18 · "答案" 13 · "题目答案是什么？" 12 · **237 distinct phrasings** in total |
| `\boxed{}` coverage | **1000 / 1000 records contain `\boxed{`**; 3 of them (#65 / #208 / #439) end with an unclosed brace — extraction needs a fallback |
| Answer shapes (rfind-last boxed) | integers/decimals 390, choice letters next (A–D single-choice alone: 109); the rest: ratios (`1:1`, `3:5`), coordinates, intervals (`-1<a<0`), chemical equations, algebraic expressions (`-2014m`), ordinals (`①②③④⑤`), etc. |
| CoT length | 104–9,059 chars, mean 679 |
| Images | 0.6 KB–1.5 MB, 140.8 MB total; two shapes: wide problem-strip screenshots (≈900×240) and full-page / camera photos (≈1080×1440); 8 multi-image records (up to 3); all 1,009 images referenced — zero orphans, zero missing |
| Content | dominated by academic problem-solving: calculus/series/functions, geometry, data structures, general knowledge, accounting practice, chemistry, etc. |

## 5. Even the official sample is raw data (tie-in with our cleaning pipeline)

Issues found while curating the sample — the input reality real data engineering deals with:

1. **14 images carry the misspelled extension `.jepg`**;
2. 43 records are **image-only with no text question** — loaders must adapt to an empty instruction; and text/vision-placeholder **order is not fixed** (#917's stem precedes the placeholder);
3. 3 records end with an **unclosed `\boxed{}` brace** (#65 / #208 / #439) — the extractor must count brace depth and handle unclosed tails;
4. **237 distinct question phrasings** — prompt normalization is necessary;
5. #35's assistant output **starts with a raw JSON blob `{"content":…`** — an upstream teacher leak passed straight into the sample;
6. #490's CoT contains the traditional-form character 「強」 — simplified/traditional and full/half-width mixing is real.

These are exactly what the [data-engineering](data-engineering_en.md) §3 cleaning & verification and §5 reward-eval answer normalization (full-width→half-width folding) handle.

## 6. Inclusion & attribution

- This directory contains only 12 images (11 records; #294 contributes two) + 1 complete JSON record (≈0.9 MB); the full package (≈142 MB unpacked) is not redistributed;
- Copyright belongs to the competition organizers; included here in small quantity for format documentation — please attribute the source when reusing.
