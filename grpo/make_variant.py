#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_variant.py —— 从已有 submit 目录生成新变体包，可选应用 Lever-A 图像尺度修正。

用法:
  python make_variant.py --src submit_Qwen3p5-0.8B-g6d-rly40-rl --tag rly40
  python make_variant.py --src submit_Qwen3p5-0.8B-g6d-rly40-rl --tag rly40mp --mp
  python make_variant.py --src submit_Qwen3p5-0.8B-g6d-rly40-rl --tag rly40bf --bf16

Lever-A (--mp): 把 processor_config.json 的 image_processor.size.longest_edge 与
preprocessor_config.json 的 size.longest_edge 同时改为 1280²=1638400，
对齐训练侧 process_sample 的「长边压到 1280」预处理，消除 9 倍 vision token 分布外。

--bf16: 修正 config.json 的 dtype 标签。RL 导出路径把 GRPO(fp32) checkpoint 的
config.json 原样拷入，写成 dtype=float32，而权重实际是 BF16 → from_pretrained
会按 fp32 加载（实测 3.41 GB vs 1.71 GB，耗时约 2 倍）。此项把三处 dtype 一并改成
bfloat16。注意：这是独立变量，与 --mp 叠加会污染单变量对比，做 A/B 时只开一个。
"""
import os, sys, json, shutil, subprocess, argparse, hashlib

G6D = "/data/g6d"
PY = "/opt/conda/envs/distill/bin/python"
TARGET = 1280 * 1280  # 1638400


def md5(p, bs=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mp", action="store_true", help="应用 Lever-A 图像尺度修正")
    ap.add_argument("--bf16", action="store_true",
                    help="把 config.json 的 dtype 标签改成 bfloat16（与实际 bf16 权重一致）")
    ap.add_argument("--keep", action="store_true", help="保留展开目录（默认打包后删除以省盘）")
    a = ap.parse_args()

    SRC = os.path.join(G6D, a.src)
    DST = os.path.join(G6D, "submit_Qwen3p5-0.8B-g6d-%s" % a.tag)
    ZIP = os.path.join(G6D, "qianwen_submit_%s_final.zip" % a.tag)
    assert os.path.isdir(SRC), "src missing: " + SRC
    if os.path.exists(DST):
        shutil.rmtree(DST)
    print("copy %s -> %s" % (SRC, DST), flush=True)
    shutil.copytree(SRC, DST)

    if a.mp:
        p = os.path.join(DST, "processor_config.json")
        c = json.load(open(p, encoding="utf-8"))
        old = dict(c["image_processor"]["size"])
        c["image_processor"]["size"]["longest_edge"] = TARGET
        json.dump(c, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("processor_config : %s -> %s" % (old, c["image_processor"]["size"]), flush=True)
        p2 = os.path.join(DST, "preprocessor_config.json")
        c2 = json.load(open(p2, encoding="utf-8"))
        old2 = dict(c2["size"])
        c2["size"]["longest_edge"] = TARGET
        json.dump(c2, open(p2, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("preprocessor_config: %s -> %s" % (old2, c2["size"]), flush=True)
        rp = os.path.join(DST, "README.md")
        if os.path.exists(rp):
            with open(rp, "a", encoding="utf-8") as f:
                f.write("\n- Lever-A 图像尺度对齐: processor/preprocessor longest_edge = 1638400 (=1280^2)\n")
    else:
        print("no-mp: 保持原配置（与 rlx30 基线同口径）", flush=True)

    # 精度标签修正：RL 导出路径把 GRPO(fp32) checkpoint 的 config.json 原样拷入，
    # 导致 dtype=float32 与 bf16 权重不符 → 加载器上采样成 fp32（显存/耗时约 2 倍）。
    if a.bf16:
        p = os.path.join(DST, "config.json")
        c = json.load(open(p, encoding="utf-8"))
        olds = []
        for key in ("dtype", "torch_dtype"):
            if key in c:
                olds.append("config.%s=%s" % (key, c[key]))
                c[key] = "bfloat16"
        for sub in ("text_config", "vision_config", "audio_config"):
            if isinstance(c.get(sub), dict):
                for key in ("dtype", "torch_dtype"):
                    if key in c[sub]:
                        olds.append("%s.%s=%s" % (sub, key, c[sub][key]))
                        c[sub][key] = "bfloat16"
        json.dump(c, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("config dtype: %s -> bfloat16" % olds, flush=True)
    else:
        print("no-bf16: config.json dtype 保持原样", flush=True)

    # 自检：回读确认
    le1 = json.load(open(os.path.join(DST, "processor_config.json"), encoding="utf-8"))["image_processor"]["size"]["longest_edge"]
    le2 = json.load(open(os.path.join(DST, "preprocessor_config.json"), encoding="utf-8"))["size"]["longest_edge"]
    print("VERIFY longest_edge processor=%s preprocessor=%s" % (le1, le2), flush=True)
    _cfg = json.load(open(os.path.join(DST, "config.json"), encoding="utf-8"))
    print("VERIFY config dtype=%s (text=%s vision=%s)" % (
        _cfg.get("dtype"), _cfg.get("text_config", {}).get("dtype"),
        _cfg.get("vision_config", {}).get("dtype")), flush=True)
    if a.mp:
        assert le1 == TARGET and le2 == TARGET, "mp patch failed"
        got = md5(os.path.join(DST, "model.safetensors"))
        print("model.safetensors md5 =", got, flush=True)

    n = len([f for f in os.listdir(DST) if os.path.isfile(os.path.join(DST, f))])
    print("files in DST =", n, flush=True)
    assert n in (13, 14), "unexpected file count %d" % n

    if os.path.exists(ZIP):
        os.remove(ZIP)
    r = subprocess.run([PY, os.path.join(G6D, "flat_zip.py"), DST, ZIP],
                       capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-400:], flush=True)
    assert r.returncode == 0, "zip failed"
    sz = os.path.getsize(ZIP)
    print("zip = %s  size = %d (%.2f GB)" % (ZIP, sz, sz / 1e9), flush=True)
    print("zip_md5 =", md5(ZIP), flush=True)
    if not a.keep:
        shutil.rmtree(DST)
        print("cleaned DST", flush=True)
    print("VARIANT_DONE %s" % a.tag, flush=True)


if __name__ == "__main__":
    main()
