#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_v14.py —— v14 导出（SFT 合并 或 RL ckpt bf16 化）
mode=sft: base(submit_rl4/v4 全量 bf16) + LoRA adapter -> merge_and_unload -> submit 目录
mode=rl : grpo checkpoint(full fp32) -> bf16 -> submit 目录 + 从 ckpt/base 补配置
mode=fixcfg: 就地修正已有 submit 目录 config.json 的 dtype 标签（不重导权重）

CLI:
  export_v14.py sft  --base <submit_dir> --adapter <lora_dir> --submit <out_dir>
  export_v14.py rl   --base <submit_dir> --ckpt  <grpo_out/checkpoint-N> --submit <out_dir> [--readme-text "..."]
  export_v14.py fixcfg --submit <dir> [--config-dtype bfloat16]

【2026-09-13 修复说明】
  现象：RL 包 config.json 里 dtype=float32，但 model.safetensors 473 个张量全是 BF16。
  实测 `AutoModelForImageTextToText.from_pretrained(dir)`（不显式给 dtype）：
      v4 包   (dtype=bfloat16) -> 加载为 torch.bfloat16，权重 1.71 GB
      rlx30 包 (dtype=float32) -> 加载为 torch.float32，权重 3.41 GB  ← 显存/耗时约 2 倍
  根因：export_rl 把 GRPO checkpoint 的 config.json 原样拷过来；GRPO 训练是全 fp32，
        该 config 里就带着 dtype=float32，与 bf16 化后的权重不符。
        （export_sft 走 model.save_pretrained，config 由 transformers 按实际精度生成，所以 v4 包正常。）
  修复：export_rl 拷完 config.json 后统一把 dtype / text_config.dtype / vision_config.dtype
        改写成实际保存精度（默认 bfloat16）。
"""
import os, sys, json, shutil, argparse, torch

G6D = "/data/g6d"

CKPT_FILES = ["config.json", "generation_config.json", "chat_template.jinja",
              "tokenizer_config.json", "tokenizer.json", "processor_config.json"]
BASE_FILES = ["vocab.json", "merges.txt", "preprocessor_config.json",
              "video_preprocessor_config.json", "special_tokens_map.json",
              "configuration.json"]


def fix_config_dtype(submit_dir, dtype="bfloat16"):
    """把 submit 目录 config.json 的精度标签改成与实际权重一致。

    同时清理嵌套的 text_config / vision_config 里的 dtype，避免加载器按子配置上采样。
    返回 (old, new) 便于日志核对；无 config.json 时返回 (None, None)。
    """
    p = os.path.join(submit_dir, "config.json")
    if not os.path.exists(p):
        return None, None
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    olds = []
    for key in ("dtype", "torch_dtype"):
        if key in cfg:
            olds.append("config.%s=%s" % (key, cfg[key]))
            cfg[key] = dtype
    for sub in ("text_config", "vision_config", "audio_config"):
        if isinstance(cfg.get(sub), dict):
            for key in ("dtype", "torch_dtype"):
                if key in cfg[sub]:
                    olds.append("%s.%s=%s" % (sub, key, cfg[sub][key]))
                    cfg[sub][key] = dtype
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return olds, dtype


def pick_adapter(adapter_dir):
    if os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        return adapter_dir
    ck = sorted([d for d in os.listdir(adapter_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    if ck:
        return os.path.join(adapter_dir, ck[-1])
    return adapter_dir


def export_sft(base, adapter, submit, config_dtype="bfloat16"):
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel
    ad = pick_adapter(adapter)
    print("adapter =", ad, flush=True)
    os.makedirs(submit, exist_ok=True)
    print("loading base bf16 ->", base, flush=True)
    bm = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.bfloat16)
    print("load LoRA + merge ...", flush=True)
    model = PeftModel.from_pretrained(bm, ad)
    model = model.merge_and_unload()
    print("saving bf16 ->", submit, flush=True)
    model.save_pretrained(submit)
    del model, bm
    for fn in sorted(os.listdir(base)):
        s = os.path.join(base, fn)
        if os.path.isfile(s) and not os.path.exists(os.path.join(submit, fn)):
            shutil.copy2(s, os.path.join(submit, fn))
            print("  copy", fn, flush=True)
    old, new = fix_config_dtype(submit, config_dtype)
    print("config dtype: %s -> %s" % (old, new), flush=True)
    print("EXPORT_DONE", flush=True)


def export_rl(base, ckpt, submit, readme_text=None, config_dtype="bfloat16"):
    from transformers import AutoModelForImageTextToText
    os.makedirs(submit, exist_ok=True)
    print("loading %s (fp32 -> bf16) ..." % ckpt, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    print("saving bf16 ->", submit, flush=True)
    model.save_pretrained(submit)
    del model
    for fn in CKPT_FILES:
        src = os.path.join(ckpt, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(submit, fn))
    for fn in BASE_FILES:
        src = os.path.join(base, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(submit, fn)):
            shutil.copy2(src, os.path.join(submit, fn))
    # 关键：ckpt 的 config.json 带的是 fp32 训练时的 dtype 标签，必须按实际保存精度改写
    old, new = fix_config_dtype(submit, config_dtype)
    print("config dtype: %s -> %s" % (old, new), flush=True)
    if readme_text:
        with open(os.path.join(submit, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_text)
    print("EXPORT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sft", "rl", "fixcfg"])
    ap.add_argument("--base", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--submit", required=True)
    ap.add_argument("--readme-text", default="")
    ap.add_argument("--config-dtype", default="bfloat16")
    a = ap.parse_args()
    if a.mode == "sft":
        export_sft(a.base, a.adapter, a.submit, a.config_dtype)
    elif a.mode == "rl":
        export_rl(a.base, a.ckpt, a.submit, a.readme_text, a.config_dtype)
    else:
        old, new = fix_config_dtype(a.submit, a.config_dtype)
        print("fixcfg: %s -> %s" % (old, new), flush=True)
        print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
