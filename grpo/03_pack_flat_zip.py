#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_pack_flat_zip.py —— 提交包平铺打包 + 自校验

平台要求 zip 内 **13 件文件平铺**（不允许任何目录层级）。原始 flat_zip.py 只在
服务器 $G6D 上留存，本文件按其行为重建，并加了打包后自校验。

用法:
  python 03_pack_flat_zip.py <submit_dir> <out.zip>
  python 03_pack_flat_zip.py --list <zip>          # 只读查看已有包清单

自校验项:
  · 件数与源目录顶层文件数一致
  · 无目录前缀（无 '/'，无空目录条目）
  · 读回 zip 逐件 CRC-32 与源文件一致
  · 打印 md5 / size，便于与基准对照
"""
import hashlib
import os
import sys
import zipfile

BASELINE = {
    "rlx30": {
        "md5": "37f81f331ed2ca75fc87d1045ab1a1b3",
        "size": 1363778296,
        "files": 13,
    }
}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        size = os.path.getsize(zip_path)
    print("zip    = %s" % zip_path)
    print("size   = %d" % size)
    print("files  = %d" % len(names))
    for n in names:
        print("  %-40s %s" % (n, "DIR!" if n.endswith("/") else ""))
    print("md5    = %s" % md5_of(zip_path))
    return names


def pack(src, out):
    if not os.path.isdir(src):
        print("FAIL: 源目录不存在 %s" % src)
        return 1
    files = sorted(
        f for f in os.listdir(src)
        if os.path.isfile(os.path.join(src, f)) and not f.startswith(".")
    )
    if not files:
        print("FAIL: 源目录为空 %s" % src)
        return 1

    # 前置：权重必须存在且尺寸合理（rlx30 实测 1,706,030,528 B）
    w = os.path.join(src, "model.safetensors")
    if not os.path.exists(w):
        print("FAIL: 缺少 model.safetensors")
        return 1
    wsz = os.path.getsize(w)
    print("model.safetensors = %d B" % wsz)
    if wsz < 1_000_000_000:
        print("WARN: 权重大小异常（<1GB），可能导出不完整")

    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in files:
            z.write(os.path.join(src, f), arcname=f)  # arcname=f ⇒ 平铺

    # ---- 自校验 ----
    src_crc = {}
    for f in files:
        with open(os.path.join(src, f), "rb") as fh:
            crc = 0
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                crc = zipfile.zlib.crc32(chunk, crc)
        src_crc[f] = crc & 0xFFFFFFFF

    with zipfile.ZipFile(tmp) as z:
        names = z.namelist()
        if len(names) != len(files):
            print("FAIL: 件数不符 zip=%d src=%d" % (len(names), len(files)))
            return 1
        bad = [n for n in names if "/" in n or n.endswith("/")]
        if bad:
            print("FAIL: 存在目录层级 %s" % bad)
            return 1
        for info in z.infolist():
            if info.file_size and info.CRC != src_crc.get(info.filename):
                print("FAIL: CRC 不符 %s" % info.filename)
                return 1
        if z.testzip() is not None:
            print("FAIL: zip 损坏")
            return 1

    if os.path.exists(out):
        os.remove(out)
    os.rename(tmp, out)

    size = os.path.getsize(out)
    print("PACK_OK files=%d size=%d md5=%s" % (len(names), size, md5_of(out)))
    b = BASELINE.get(os.path.basename(src).replace("submit_Qwen3p5-0.8B-g6d-", "").replace("-rl", ""))
    if b:
        print("baseline: files=%d size=%d md5=%s" % (b["files"], b["size"], b["md5"]))
        print("note: md5 不必相等（README 文本可能不同）；件数与 size 应一致（±1KB）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--list":
        list_zip(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(pack(sys.argv[1], sys.argv[2]))
